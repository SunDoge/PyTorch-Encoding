# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Hang Zhang
# Email: zhanghang0704@gmail.com
# Copyright (c) 2020
##
# LICENSE file in the root directory of this source tree
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import argparse
import logging
import os
import random
from collections import OrderedDict

import encoding
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from encoding.nn import LabelSmoothing, NLLMultiLabelSmooth
from encoding.utils import accuracy
from mmcv.runner import DistSamplerSeedHook, Runner, get_dist_info, init_dist
from torch.nn.parallel import DistributedDataParallel


class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='Deep Encoding')
        parser.add_argument('--dataset', type=str, default='imagenet',
                            help='training dataset (default: cifar10)')
        parser.add_argument('--base-size', type=int, default=None,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=224,
                            help='crop image size')
        parser.add_argument('--label-smoothing', type=float, default=0.1,
                            help='label-smoothing (default eta: 0.0)')
        parser.add_argument('--mixup', type=float, default=0,
                            help='mixup (default eta: 0.0)')
        parser.add_argument('--rand-aug', action='store_true',
                            default=False, help='random augment')
        # model params
        parser.add_argument('--model', type=str, default='resnet50',
                            help='network model type (default: densenet)')
        parser.add_argument('--rectify', action='store_true',
                            default=False, help='rectify convolution')
        parser.add_argument('--rectify-avg', action='store_true',
                            default=False, help='rectify convolution')
        parser.add_argument('--pretrained', action='store_true',
                            default=False, help='load pretrianed mode')
        parser.add_argument('--last-gamma', action='store_true', default=False,
                            help='whether to init gamma of the last BN layer in \
                            each bottleneck to 0 (default: False)')
        parser.add_argument('--dropblock-prob', type=float, default=0,
                            help='DropBlock prob. default is 0.')
        parser.add_argument('--final-drop', type=float, default=0,
                            help='final dropout prob. default is 0.')
        # training params
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='batch size for training (default: 128)')
        parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                            help='batch size for testing (default: 256)')
        parser.add_argument('--epochs', type=int, default=270, metavar='N',
                            help='number of epochs to train (default: 600)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='the epoch number to start (default: 1)')
        parser.add_argument('--workers', type=int, default=4,
                            metavar='N', help='dataloader threads')
        # optimizer
        parser.add_argument('--lr', type=float, default=0.025, metavar='LR',
                            help='learning rate (default: 0.1)')
        parser.add_argument('--lr-scheduler', type=str, default='cos',
                            help='learning rate scheduler (default: cos)')
        parser.add_argument('--warmup-epochs', type=int, default=5,
                            help='number of warmup epochs (default: 0)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4,
                            metavar='M', help='SGD weight decay (default: 1e-4)')
        parser.add_argument('--no-bn-wd', action='store_true',
                            default=False, help='no bias decay')
        # seed
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')
        # distributed
        parser.add_argument('--world-size', default=1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--local_rank', default=0, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(args.local_rank)
        return args


def get_root_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def batch_processor(model, data, train_mode, criterion, mixup):
    if train_mode:
        mixup = False
    img, label = data
    if not mixup:
        label = label.cuda(non_blocking=True)
    pred = model(img)
    loss = criterion(pred, label)
    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()
    if not train_mode:
        acc_top1, acc_top5 = accuracy(pred, label, topk=(1, 5))
        log_vars['acc_top1'] = acc_top1.item()
        log_vars['acc_top5'] = acc_top5.item()
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=img.size(0))
    return outputs


class MixUpWrapper(object):
    def __init__(self, alpha, num_classes, dataloader, device):
        self.alpha = alpha
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.device = device
        self.sampler = dataloader.sampler

    def mixup_loader(self, loader):
        def mixup(alpha, num_classes, data, target):
            with torch.no_grad():
                bs = data.size(0)
                c = np.random.beta(alpha, alpha)
                perm = torch.randperm(bs).cuda()

                md = c * data + (1-c) * data[perm, :]
                mt = c * target + (1-c) * target[perm, :]
                return md, mt

        for input, target in loader:
            input, target = input.cuda(self.device), target.cuda(self.device)
            target = torch.nn.functional.one_hot(target, self.num_classes)
            i, t = mixup(self.alpha, self.num_classes, input, target)
            yield i, t

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self.mixup_loader(self.dataloader)


def main():
    args = Options().parse()
    torch.backends.cudnn.benchmark = True
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    init_dist('pytorch', backend=args.dist_backend)
    logger = get_root_logger('INFO')
    args.lr = args.lr * dist.get_world_size()
    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)
    main_worker(args)


# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []


def torch_dist_sum(gpu, *args):
    process_group = torch.distributed.group.WORLD
    tensor_args = []
    pending_res = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tensor_arg = arg.clone().reshape(1).detach().cuda(gpu)
        else:
            tensor_arg = torch.tensor(arg).reshape(1).cuda(gpu)
        tensor_args.append(tensor_arg)
        pending_res.append(torch.distributed.all_reduce(
            tensor_arg, group=process_group, async_op=True))
    for res in pending_res:
        res.wait()
    return tensor_args


def main_worker(args):
    args.gpu = args.local_rank
    # args.rank = args.rank * ngpus_per_node + gpu
    print('rank: {} / {}'.format(args.local_rank, dist.get_world_size()))
    # init the args
    global best_pred, acclist_train, acclist_val

    if args.gpu == 0:
        print(args)

    # init dataloader
    transform_train, transform_val = encoding.transforms.get_transform(
        args.dataset, args.base_size, args.crop_size, args.rand_aug)
    trainset = encoding.datasets.get_dataset(
        args.dataset, root=os.path.expanduser('~/data'),
        transform=transform_train, train=True, download=True)
    valset = encoding.datasets.get_dataset(
        args.dataset, root=os.path.expanduser('~/data'),
        transform=transform_val, train=False, download=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=train_sampler)

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        valset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=val_sampler)
    print(len(train_loader), len(val_loader))
    # init the model
    model_kwargs = {}
    if args.pretrained:
        model_kwargs['pretrained'] = True

    if args.final_drop > 0.0:
        model_kwargs['final_drop'] = args.final_drop

    if args.dropblock_prob > 0.0:
        model_kwargs['dropblock_prob'] = args.dropblock_prob

    if args.last_gamma:
        model_kwargs['last_gamma'] = True

    if args.rectify:
        model_kwargs['rectified_conv'] = True
        model_kwargs['rectify_avg'] = args.rectify_avg

    model = encoding.models.get_model(args.model, **model_kwargs)

    if args.dropblock_prob > 0.0:
        from functools import partial
        from encoding.nn import reset_dropblock
        nr_iters = (args.epochs - args.warmup_epochs) * len(train_loader)
        apply_drop_prob = partial(
            reset_dropblock, args.warmup_epochs * len(train_loader),
            nr_iters, 0.0, args.dropblock_prob)
        model.apply(apply_drop_prob)

    if args.gpu == 0:
        print(model)

    if args.mixup > 0:
        train_loader = MixUpWrapper(args.mixup, 1000, train_loader, args.gpu)
        criterion = NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        criterion = LabelSmoothing(args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    criterion.cuda(args.gpu)
    model = DistributedDataParallel(
        model.cuda(), device_ids=[torch.cuda.current_device()])

    # criterion and optimizer
    if args.no_bn_wd:
        parameters = model.named_parameters()
        param_dict = {}
        for k, v in parameters:
            param_dict[k] = v
        bn_params = [v for n, v in param_dict.items() if (
            'bn' in n or 'bias' in n)]
        rest_params = [v for n, v in param_dict.items() if not (
            'bn' in n or 'bias' in n)]
        if args.gpu == 0:
            print(" Weight decay NOT applied to BN parameters ")
            print(
                f'len(parameters): {len(list(model.parameters()))} = {len(bn_params)} + {len(rest_params)}')
        optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0},
                                     {'params': rest_params, 'weight_decay': args.weight_decay}],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # scheduler = LR_Scheduler(args.lr_scheduler,
    #                          base_lr=args.lr,
    #                          num_epochs=args.epochs,
    #                          iters_per_epoch=len(train_loader),
    #                          warmup_epochs=args.warmup_epochs)
    directory = "runs/%s/%s/%s/" % (args.dataset, args.model, args.checkname)

    runner = Runner(
        model,
        batch_processor,
        optimizer,
        directory,
        log_level='INFO')

    lr_config = dict(
        policy='cosine',
        warmup_ratio=0.01,
        warmup='linear',
        warmup_iters=len(train_loader)*args.warmup_epochs)

    log_config = dict(
        interval=20,
        hooks=[
            dict(type='TextLoggerHook'),
            dict(type='TensorboardLoggerHook')
        ])

    runner.register_training_hooks(
        lr_config=lr_config,
        optimizer_config=dict(grad_clip=dict(max_norm=40, norm_type=2)),
        checkpoint_config=dict(interval=5),
        log_config=log_config)

    runner.register_hook(DistSamplerSeedHook())
    if args.resume is not None:
        runner.resume(args.resume)

    runner.run([train_loader, val_loader],
               [('train', 1), ('val', 1)],
               args.epochs,
               criterion=criterion,
               mixup=args.mixup)


if __name__ == "__main__":
    main()
