#!/bin/bash
source /etc/network_turbo
cd /root/autodl-tmp/Ryan/cst_antiuav/experiments/tcs_ablation_20261017
python train_ablation.py --name tcs_yolo8n > logs_tcs_yolo8n.log 2>&1 &
W8=$!
python train_ablation.py --name tcs_far15_yolo11n > logs_tcs_far15.log 2>&1 &
F15=$!
wait $W8
echo "[$(date -Is)] yolo8n done -> yolo11s"
python train_ablation.py --name tcs_yolo11s > logs_tcs_yolo11s.log 2>&1
echo "[$(date -Is)] yolo11s done -> far20"
python train_ablation.py --name tcs_far20_yolo11n > logs_tcs_far20.log 2>&1
wait $F15
echo "[$(date -Is)] far15 done"
echo "TCS_ABLATION_DONE"
