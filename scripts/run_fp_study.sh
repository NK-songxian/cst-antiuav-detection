#!/bin/bash
# FP study driver: 4 groups, 2 parallel waves (G1+G2, then G3+G4), then sequential retry for failures.
cd /root/autodl-tmp/Ryan/cst_antiuav || exit 1
source /root/miniconda3/etc/profile.d/conda.sh
LOG=runs/fp_study/driver.log
mkdir -p runs/fp_study

echo "[driver] wave 1: G1 + G2 start $(date)" >> "$LOG"
python scripts/train_fp_study.py --group G1 > runs/fp_study/G1.log 2>&1 &
P1=$!
python scripts/train_fp_study.py --group G2 > runs/fp_study/G2.log 2>&1 &
P2=$!
wait $P1 $P2
echo "[driver] wave 1 done $(date)" >> "$LOG"

echo "[driver] wave 2: G3 + G4 start $(date)" >> "$LOG"
python scripts/train_fp_study.py --group G3 > runs/fp_study/G3.log 2>&1 &
P3=$!
python scripts/train_fp_study.py --group G4 > runs/fp_study/G4.log 2>&1 &
P4=$!
wait $P3 $P4
echo "[driver] wave 2 done $(date)" >> "$LOG"

# sequential retry for any failed group (e.g. OOM when sharing GPU)
for g in G1 G2 G3 G4; do
  if [ ! -f "runs/fp_study/$g/weights/best.pt" ]; then
    echo "[driver] retry $g sequentially $(date)" >> "$LOG"
    python scripts/train_fp_study.py --group "$g" > "runs/fp_study/$g.log" 2>&1
  fi
done

touch runs/fp_study/FP_STUDY_DONE
echo "[driver] FP_STUDY_DONE $(date)" >> "$LOG"
