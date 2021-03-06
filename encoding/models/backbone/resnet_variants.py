"""ResNet variants"""

import torch

from ..model_store import get_model_file
from .resnet import Bottleneck, ResNet

__all__ = ['resnet50s', 'resnet101s', 'resnet152s',
           'resnet50d', 'resnet50d_avdfast', 'resnet50d_avd',
           'resnet50_avgdown',
           'resnet50_avgdown_avdfast', 'resnet50_avgdown_avd']


# pspnet version of ResNet
def resnet50s(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNetS-50 model as in PSPNet.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['deep_stem'] = True
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet50s', root=root)), strict=False)
    return model


def resnet101s(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNetS-101 model as in PSPNet.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['deep_stem'] = True
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet101s', root=root)), strict=False)
    return model


def resnet152s(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNetS-152 model as in PSPNet.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['deep_stem'] = True
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet152s', root=root)), strict=False)
    return model

# ResNet-D
def resnet50d(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   deep_stem=True, stem_width=32,
                   avg_down=True, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet50d', root=root)), strict=False)
    return model

# ResNet-D-avd
def resnet50d_avdfast(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=True, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet50d_avdfast', root=root)), strict=False)
    return model


def resnet50d_avd(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet50d_avd', root=root)), strict=False)
    return model


def resnet50_avgdown(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNetS-50 model as in PSPNet.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], avg_down=True, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet50_avgdown', root=root)), strict=False)
    return model


def resnet50_avgdown_avdfast(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNetS-50 model as in PSPNet.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], avg_down=True, 
                   avd=True, avd_first=True, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet50_avgdown_avdfast', root=root)), strict=False)
    return model


def resnet50_avgdown_avd(pretrained=False, root='~/.encoding/models', **kwargs):
    """Constructs a ResNetS-50 model as in PSPNet.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(
            get_model_file('resnet50_avgdown_avd', root=root)), strict=False)
    return model
