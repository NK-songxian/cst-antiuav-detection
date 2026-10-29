#!/bin/bash
LOG=/root/autodl-tmp/Ryan/cst_antiuav/logs/smoke_train.log
# 等环境安装结束（最长40分钟）
for i in $(seq 1 80); do
    pgrep -f setup_env.sh >/dev/null || break
    sleep 30
done
source /root/miniconda3/etc/profile.d/conda.sh && conda activate cst
PY=/root/miniconda3/envs/cst/bin/python
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('CUDA OK:', torch.cuda.get_device_name(0))" || { echo "ENV_VERIFY_FAIL"; exit 1; }
cd /root/autodl-tmp/Ryan/cst_antiuav
$PY -m ultralytics yolo detect train \
    model=yolo11n.pt data=data/yolo/cst.yaml \
    epochs=1 fraction=0.1 imgsz=640 batch=32 device=0 workers=8 \
    project=runs name=smoke exist_ok=True 2>&1 | tail -30
echo "SMOKE_DONE"
