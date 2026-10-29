#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda env list | grep -q "^cst " || conda create -y -n cst python=3.11
P=/root/miniconda3/envs/cst/bin/pip
# torch/torchvision 走阿里云 pytorch-wheels cu128 镜像（Blackwell 需 cu128），依赖走 aliyun pypi
$P install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
    --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/ \
    --index-url https://mirrors.aliyun.com/pypi/simple/
$P install ultralytics opencv-python-headless pyyaml matplotlib \
    --index-url https://mirrors.aliyun.com/pypi/simple/
/root/miniconda3/envs/cst/bin/python -c "import torch,ultralytics; print('OK torch',torch.__version__,'cuda',torch.cuda.is_available())"
