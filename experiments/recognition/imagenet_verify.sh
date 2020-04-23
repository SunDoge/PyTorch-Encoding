#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

# resnet50  ../../gluonvision_pretrain/resnet50-0ef8ed2d.pth 
$PYTHON verify.py --dataset imagenet --model $1  --crop-size 224 --verify $2  ${@:4}