#!/usr/bin/env python3
"""前提验证：模型是否在目标离开后的原位置（热点残留）误报？
对 val 背景帧（无GT），用 baseline best.pt 预测：
  1) 统计 FP 率（conf>0.25 / >0.1）
  2) FP 框与"同序列最近正样本帧 GT 框"的 IoU 集中度
  3) 对照组：与随机其他序列 GT 框的 IoU（排除大小巧合）
  4) 按目标离开的时间距离分桶看 FP 率
"""
import json, random
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import torch, numpy as np

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
VAL = ROOT/'data/yolo/val'
IDX = json.loads((ROOT/'data/yolo/attribute_index.json').read_text())['val']

# 按序列组织帧
seqs = defaultdict(dict)          # seq -> frame -> (stem, exist)
for stem, rec in IDX.items():
    seqs[rec['seq']][rec['frame']] = (stem, rec['exist'], rec['attr'])

# GT 框（像素）
def gt_boxes(stem):
    lbl = VAL/'labels'/f'{stem}.txt'
    if not lbl.exists(): return []
    out = []
    for l in lbl.read_text().splitlines():
        if not l.strip(): continue
        p = l.split()
        cx, cy, w, h = (float(v)*s for v, s in zip(p[1:5], (640,512,640,512)))
        out.append((cx-w/2, cy-h/2, cx+w/2, cy+h/2))
    return out

def last_gt(seq, frame, max_gap=200):
    """同序列 frame 之前最近的 exist=1 帧的 GT"""
    for d in range(5, max_gap):
        cand = [f for f in seqs[seq] if f < frame and frame - d <= f < frame - d + 5]
        for f in sorted(cand, reverse=True):
            s, e, a = seqs[seq][f]
            if e == 1:
                b = gt_boxes(s)
                if b: return f, b
    return None, []

def iou(b, boxes):
    x1,y1,x2,y2 = b
    best = 0.0
    for gx1,gy1,gx2,gy2 in boxes:
        iw = max(0, min(x2,gx2)-max(x1,gx1)); ih = max(0, min(y2,gy2)-max(y1,gy1))
        inter = iw*ih; union = (x2-x1)*(y2-y1)+(gx2-gx1)*(gy2-gy1)-inter
        best = max(best, inter/union if union > 0 else 0)
    return best

# 背景帧列表（exist=0 且同序列之前存在正样本帧）
bg_frames = []
for stem, rec in IDX.items():
    if rec['exist'] == 0:
        prev = [f for f in seqs[rec['seq']] if f < rec['frame'] and seqs[rec['seq']][f][1] == 1]
        if prev:
            bg_frames.append((rec['seq'], rec['frame'], stem))
print(f'背景帧（且此前有目标出现）: {len(bg_frames)}', flush=True)

model = YOLO(str(ROOT/'runs/baseline_yolo11n/weights/best.pt'))
paths = [VAL/'images'/f'{s}.jpg' for _,_,s in bg_frames]
preds = {}
for i in range(0, len(paths), 512):
    for r in model.predict(paths[i:i+512], conf=0.05, imgsz=640, batch=16, stream=True, verbose=False):
        preds[Path(r.path).stem] = (r.boxes.xyxy.tolist(), r.boxes.conf.tolist())
    torch.cuda.empty_cache()

# 全数据集随机正样本框做对照
all_gt = [b for stem, rec in IDX.items() if rec['exist']==1 for b in gt_boxes(stem)]
rng = random.Random(0)

buckets = defaultdict(lambda: [0,0,0,0.0,0.0])  # gap桶: [n, n_fp25, n_fp10, sum_iou_last, sum_iou_rand]
for seq, frame, stem in bg_frames:
    dets, confs = preds.get(stem, ([], []))
    if not dets: 
        continue
    lf, boxes = last_gt(seq, frame)
    gap = frame - lf if lf else 9999
    b = buckets[min(gap//20*20, 200) if gap<200 else 200]
    b[0] += 1
    ious_last = [iou(d, boxes) for d in dets] if boxes else [0.0]*len(dets)
    ious_rand = [iou(d, [rng.choice(all_gt)]) for d in dets]
    c25 = [c for c in confs if c > 0.25]; c10 = [c for c in confs if c > 0.10]
    if c25: b[1] += 1
    if c10: b[2] += 1
    # 只看强FP（conf>0.25）的空间位置
    strong = [v for v, c in zip(ious_last, confs) if c > 0.25]
    rands  = [v for v, c in zip(ious_rand, confs) if c > 0.25]
    if strong:
        b[3] += max(strong); b[4] += max(rands)

print(f"\n{'gap(帧)':>8} {'n':>5} {'FP@.25':>7} {'FP@.10':>7} {'旧位置IoU':>9} {'随机IoU':>8}")
tot = [0,0,0]
for g in sorted(buckets):
    n, f25, f10, si, sr = buckets[g]
    tot[0]+=n; tot[1]+=f25; tot[2]+=f10
    if n:
        print(f"{g:>8} {n:>5} {f25/max(n,1):>7.3f} {f10/max(n,1):>7.3f} {si/n:>9.4f} {sr/n:>8.4f}")
print(f"{'TOTAL':>8} {tot[0]:>5} {tot[1]/max(tot[0],1):>7.3f} {tot[2]/max(tot[0],1):>7.3f}")
