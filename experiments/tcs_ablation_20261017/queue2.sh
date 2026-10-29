#!/bin/bash
# remaining jobs with 2-slot packing; yolo8n already running externally
cd /root/autodl-tmp/Ryan/cst_antiuav/experiments/tcs_ablation_20261017
python build_data.py > logs/build_data2.log 2>&1
echo "[$(date -Is)] far data built, start far15 (slot2, yolo8n occupies slot1)"
python train_ablation.py --name tcs_far15_yolo11n > logs_tcs_far15.log 2>&1 &
F15=$!
( while [ ! -f runs/tcs_yolo8n/tcs_yolo8n.result.json ]; do sleep 120; done
  echo "[$(date -Is)] yolo8n done, start yolo11s"
  python train_ablation.py --name tcs_yolo11s > logs_tcs_yolo11s.log 2>&1
  echo "[$(date -Is)] yolo11s done, start far20"
  python train_ablation.py --name tcs_far20_yolo11n > logs_tcs_far20.log 2>&1 ) &
CHAIN=$!
wait $F15; S1=$?
wait $CHAIN; S2=$?
echo "TCS_ABLATION_DONE far15=$S1 chain=$S2"
