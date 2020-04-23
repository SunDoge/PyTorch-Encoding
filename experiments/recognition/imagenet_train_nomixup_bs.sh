#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$2 train_dist_mmcv.py --model $1 \
--checkname nomixup  --no-bn-wd --last-gamma  --resume $3 --batch-size 96 --epochs 120 ${@:6}

# without
# --rand-aug
# --mixup 0.2