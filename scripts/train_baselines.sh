#!/bin/bash
# 三个基线模型顺序训练（共享GPU；batch=256 充分利用空闲显存加速）
cd /root/autodl-tmp/Ryan/cst_antiuav
for MODEL in yolo11n yolo11s yolov8n; do
    W=/root/autodl-tmp/Ryan/cst_antiuav/${MODEL}.pt
    [ -f "$W" ] || { echo "MISSING WEIGHTS $W, skip"; continue; }
    echo "===== TRAIN $MODEL $(date '+%T') ====="
    PYTHONUNBUFFERED=1 python3 - << PYEOF
from ultralytics import YOLO
m = YOLO('$W')
m.train(data='/root/autodl-tmp/Ryan/cst_antiuav/data/yolo/cst.yaml',
        epochs=100, patience=20, imgsz=640, batch=256, device=0, workers=16,
        project='/root/autodl-tmp/Ryan/cst_antiuav/runs', name='baseline_$MODEL', exist_ok=True)
PYEOF
done
echo "ALL_BASELINES_DONE $(date '+%T')"
