#!/usr/bin/env bash
# Sequential TCE seed0 queue: T3 first (main hypothesis), then T2.
# GPU runs require launching outside the sandbox (CUDA init fails inside).
set -u
STUDY=/root/autodl-tmp/Ryan/cst_antiuav/experiments/tce_revision_20261109
cd "$STUDY"
: > queue.log
python3 train_tce.py --arm t3 --seed 0 >> queue.log 2>&1 \
  && python3 audit_tce.py --arm t3 --seed 0 --split val >> queue.log 2>&1 \
  && python3 audit_tce.py --arm t3 --seed 0 --split test >> queue.log 2>&1
echo "T3_STAGE_EXIT=$?" >> queue.log
python3 train_tce.py --arm t2 --seed 0 >> queue.log 2>&1 \
  && python3 audit_tce.py --arm t2 --seed 0 --split val >> queue.log 2>&1 \
  && python3 audit_tce.py --arm t2 --seed 0 --split test >> queue.log 2>&1
echo "T2_STAGE_EXIT=$?" >> queue.log
echo "TCE_QUEUE_DONE" >> queue.log
