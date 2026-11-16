#!/bin/bash
# TCE v2 seed0: ghost regression-exclusion training + frozen-protocol audits.
cd /root/autodl-tmp/Ryan/cst_antiuav/experiments/tce_revision_20261109
Q=queue_v2.log
: > $Q
export PYTHONUNBUFFERED=1

python3 train_tce.py --arm t3v2 --seed 0 >> $Q 2>&1 \
  && python3 audit_tce.py --arm t3v2 --seed 0 --split val >> $Q 2>&1 \
  && python3 audit_tce.py --arm t3v2 --seed 0 --split test >> $Q 2>&1
echo "T3V2_STAGE_EXIT=$?" >> $Q
echo "TCE_QUEUE_V2_DONE" >> $Q
