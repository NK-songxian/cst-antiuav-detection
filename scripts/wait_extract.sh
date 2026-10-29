#!/bin/bash
# 监视 zip 上传：大小连续 3 分钟不变视为完成，然后解压
ZIP=/root/autodl-tmp/Ryan/cst_antiuav/data/CST-AntiUAV.zip
OUT=/root/autodl-tmp/Ryan/cst_antiuav/data
LOG=/root/autodl-tmp/Ryan/cst_antiuav/logs/wait_extract.log

STABLE=0
PREV=-1
while true; do
    CUR=$(stat -c %s "$ZIP" 2>/dev/null || echo 0)
    echo "$(date '+%F %T') size=${CUR}" >> "$LOG"
    if [ "$CUR" = "$PREV" ]; then
        STABLE=$((STABLE+1))
    else
        STABLE=0
    fi
    if [ "$STABLE" -ge 3 ]; then
        echo "$(date '+%F %T') upload stable at ${CUR} bytes, extracting..." >> "$LOG"
        unzip -q -o "$ZIP" -d "$OUT" >> "$LOG" 2>&1
        if [ $? -eq 0 ]; then
            echo "$(date '+%F %T') EXTRACT_OK" >> "$LOG"
        else
            echo "$(date '+%F %T') EXTRACT_FAIL (zip may be incomplete/corrupt)" >> "$LOG"
        fi
        exit 0
    fi
    PREV=$CUR
    sleep 60
done
