#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$2 train_dist_mmcv.py --model $1 \
--checkname mixup  --no-bn-wd --last-gamma  --rand-aug  --mixup 0.2  --epochs 270 ${@:7}

# without
# --rand-aug
# --mixup 0.2
# --epochs 270