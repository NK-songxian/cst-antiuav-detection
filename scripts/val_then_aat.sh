#!/bin/bash
# 1) baseline_yolo11n best.pt 正式验证  2) AAT-v1 训练（属性先验重采样版）
cd /root/autodl-tmp/Ryan/cst_antiuav
echo "===== VAL baseline_yolo11n $(date '+%T') ====="
PYTHONUNBUFFERED=1 python3 - << 'PYEOF'
from ultralytics import YOLO
m = YOLO('/root/autodl-tmp/Ryan/cst_antiuav/runs/baseline_yolo11n/weights/best.pt')
m.val(data='/root/autodl-tmp/Ryan/cst_antiuav/data/yolo/cst.yaml', batch=256, workers=12,
      project='/root/autodl-tmp/Ryan/cst_antiuav/runs', name='val_yolo11n', exist_ok=True)
PYEOF
echo "===== TRAIN AAT_v1 (yolo11n, oversampled) $(date '+%T') ====="
PYTHONUNBUFFERED=1 python3 - << 'PYEOF'
from ultralytics import YOLO
m = YOLO('/root/autodl-tmp/Ryan/cst_antiuav/yolo11n.pt')
m.train(data='/root/autodl-tmp/Ryan/cst_antiuav/data/yolo/cst.yaml',
        epochs=60, patience=15, imgsz=640, batch=256, device=0, workers=12,
        project='/root/autodl-tmp/Ryan/cst_antiuav/runs', name='AAT_v1_yolo11n', exist_ok=True)
PYEOF
echo "===== VAL AAT_v1 $(date '+%T') ====="
PYTHONUNBUFFERED=1 python3 - << 'PYEOF'
from ultralytics import YOLO
m = YOLO('/root/autodl-tmp/Ryan/cst_antiuav/runs/AAT_v1_yolo11n/weights/best.pt')
m.val(data='/root/autodl-tmp/Ryan/cst_antiuav/data/yolo/cst.yaml', batch=256, workers=12,
      project='/root/autodl-tmp/Ryan/cst_antiuav/runs', name='val_AAT_v1', exist_ok=True)
PYEOF
echo "CHAIN_DONE $(date '+%T')"
