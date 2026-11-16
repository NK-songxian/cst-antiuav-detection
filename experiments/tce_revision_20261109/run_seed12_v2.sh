#!/bin/bash
# TCE v2 seeds 1-2: ghost regression-exclusion training + frozen-protocol audits.
cd /root/autodl-tmp/Ryan/cst_antiuav/experiments/tce_revision_20261109
Q=queue_v2_s12.log
: > $Q
export PYTHONUNBUFFERED=1

for S in 1 2; do
  python3 train_tce.py --arm t3v2 --seed $S >> $Q 2>&1 \
    && python3 audit_tce.py --arm t3v2 --seed $S --split val >> $Q 2>&1 \
    && python3 audit_tce.py --arm t3v2 --seed $S --split test >> $Q 2>&1
  echo "T3V2_SEED${S}_STAGE_EXIT=$?" >> $Q
done
echo "TCE_QUEUE_V2_S12_DONE" >> $Q
