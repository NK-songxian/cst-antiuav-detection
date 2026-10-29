#!/bin/bash
# TCS ablation queue: wave1 = tcs_yolo8n + tcs_delta1_yolo11n (parallel),
# wave2 = tcs_yolo11s + tcs_delta10_yolo11n (parallel). 100e / patience 20 / batch 256.
cd /root/autodl-tmp/Ryan/cst_antiuav/experiments/tcs_ablation_20261017
echo "[$(date -Is)] wave1 start: tcs_yolo8n + tcs_delta1_yolo11n"
python train_ablation.py --name tcs_yolo8n        > logs_tcs_yolo8n.log 2>&1 &
P1=$!
python train_ablation.py --name tcs_delta1_yolo11n > logs_tcs_delta1.log 2>&1 &
P2=$!
wait $P1; S1=$?
wait $P2; S2=$?
echo "[$(date -Is)] wave1 done: yolo8n=$S1 delta1=$S2"
echo "[$(date -Is)] wave2 start: tcs_yolo11s + tcs_delta10_yolo11n"
python train_ablation.py --name tcs_yolo11s       > logs_tcs_yolo11s.log 2>&1 &
P3=$!
python train_ablation.py --name tcs_delta10_yolo11n > logs_tcs_delta10.log 2>&1 &
P4=$!
wait $P3; S3=$?
wait $P4; S4=$?
echo "[$(date -Is)] wave2 done: yolo11s=$S3 delta10=$S4"
echo "TCS_ABLATION_DONE yolo8n=$S1 delta1=$S2 yolo11s=$S3 delta10=$S4"
