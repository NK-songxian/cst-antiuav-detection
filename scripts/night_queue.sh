#!/bin/bash
# 夜间队列：AAT-v2（仅TC×2）→ 正式验证 → yolo11s 基线 → yolov8n 基线
cd /root/autodl-tmp/Ryan/cst_antiuav
echo "===== TRAIN AAT_v2 (TC only) $(date '+%T') ====="
PYTHONUNBUFFERED=1 python3 - << 'PYEOF'
from ultralytics import YOLO
m = YOLO('/root/autodl-tmp/Ryan/cst_antiuav/yolo11n.pt')
m.train(data='/root/autodl-tmp/Ryan/cst_antiuav/data/yolo/cst.yaml',
        epochs=60, patience=15, imgsz=640, batch=256, device=0, workers=12,
        project='/root/autodl-tmp/Ryan/cst_antiuav/runs', name='AAT_v2_yolo11n', exist_ok=True)
PYEOF
echo "===== VAL AAT_v2 $(date '+%T') ====="
PYTHONUNBUFFERED=1 python3 - << 'PYEOF'
from ultralytics import YOLO
m = YOLO('/root/autodl-tmp/Ryan/cst_antiuav/runs/AAT_v2_yolo11n/weights/best.pt')
m.val(data='/root/autodl-tmp/Ryan/cst_antiuav/data/yolo/cst.yaml', batch=256, workers=12,
      project='/root/autodl-tmp/Ryan/cst_antiuav/runs', name='val_AAT_v2', exist_ok=True)
PYEOF
for MODEL in yolo11s yolov8n; do
    echo "===== TRAIN baseline $MODEL $(date '+%T') ====="
    PYTHONUNBUFFERED=1 python3 - << PYEOF
from ultralytics import YOLO
m = YOLO('/root/autodl-tmp/Ryan/cst_antiuav/${MODEL}.pt')
m.train(data='/root/autodl-tmp/Ryan/cst_antiuav/data/yolo/cst.yaml',
        epochs=100, patience=20, imgsz=640, batch=256, device=0, workers=12,
        project='/root/autodl-tmp/Ryan/cst_antiuav/runs', name='baseline_$MODEL', exist_ok=True)
PYEOF
done
echo "NIGHT_QUEUE_DONE $(date '+%T')"
