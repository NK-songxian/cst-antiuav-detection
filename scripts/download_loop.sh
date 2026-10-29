#!/bin/bash
source /etc/network_turbo 2>/dev/null
cd /root/autodl-tmp/Ryan/cst_antiuav/data
F1='https://drive.google.com/drive/folders/13DIRzt1pC2c4Dhyi6yFhHBeMFoKPQtxU'
F2='https://drive.google.com/drive/folders/132CBS47zNjQemhBXGKTEsHdTgwiGzMub'
for i in $(seq 1 40); do
  echo "[round $i] trying F1 ..." ; /root/miniconda3/bin/gdown --folder "$F1" --continue --retries 3 && echo F1_DONE && break
  echo "[round $i] F1 failed, trying F2 ..."
  /root/miniconda3/bin/gdown --folder "$F2" --continue --retries 3 && echo F2_DONE && break
  sleep 180
done
echo LOOP_EXIT
