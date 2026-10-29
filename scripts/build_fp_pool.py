#!/usr/bin/env python3
"""第1步：错题池构建
baseline best.pt 在 train exist=0 帧上挖高置信误报：
  - 按序列限流(20) + 同热点去重(IoU>0.3留最高分) + 时间间隔>=25原始帧
  - 记录几何/对比度统计（d, r, c, 极性）
  - 可疑档（疑似漏标真目标）单独导出 crop 供人工检查
输出: runs/fp_pool/pool.json + suspicious/*.jpg + 控制台摘要
"""
import json, math
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import torch

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
TRAIN = ROOT/'data/yolo/train'
IDX = json.loads((ROOT/'data/yolo/attribute_index.json').read_text())['train']
OUT = ROOT/'runs/fp_pool'; OUT.mkdir(parents=True, exist_ok=True)
(OUT/'suspicious').mkdir(exist_ok=True)

CONF_MIN = 0.10          # 弱FP收集线（强FP另记 conf>0.25）
CAP_PER_SEQ = 20
MIN_GAP = 25             # 原始帧间隔
HOTSPOT_IOU = 0.3

seqs = defaultdict(dict)
for stem, rec in IDX.items():
    seqs[rec['seq']][rec['frame']] = (stem, rec['exist'])
neg_frames = sorted((rec['seq'], rec['frame'], stem) for stem, rec in IDX.items() if rec['exist'] == 0)
print(f'train exist=0 帧: {len(neg_frames)}', flush=True)

model = YOLO(str(ROOT/'runs/baseline_yolo11n/weights/best.pt'))
paths = [TRAIN/'images'/f'{s}.jpg' for _,_,s in neg_frames]
raw = []   # (seq, frame, stem, box, conf)
for i in range(0, len(paths), 512):
    for r in model.predict(paths[i:i+512], conf=CONF_MIN, imgsz=640, batch=16, stream=True, verbose=False):
        seq, frame, stem = next((s,f,st) for st in [Path(r.path).stem] for s,f,st in [next(n for n in neg_frames if n[2]==st)])
        for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            raw.append((seq, frame, stem, b, c))
    torch.cuda.empty_cache()
print(f'原始 FP 检测: {len(raw)} (conf>={CONF_MIN})', flush=True)

def iou(a, b):
    iw = max(0, min(a[2],b[2])-max(a[0],b[0])); ih = max(0, min(a[3],b[3])-max(a[1],b[1]))
    inter = iw*ih; union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union > 0 else 0

def contrast(img, b):
    """归一化有符号局部对比 c=(mean_core-mean_ring)/(sigma_ring+eps)"""
    x1,y1,x2,y2 = [int(round(v)) for v in b]
    H,W = img.shape
    cx1,cy1,cx2,cy2 = max(0,x1),max(0,y1),min(W,x2),min(H,y2)
    core = img[cy1:cy2, cx1:cx2]
    m = max(2, int(round(max(x2-x1, y2-y1))))
    rx1,ry1,rx2,ry2 = max(0,x1-m),max(0,y1-m),min(W,x2+m),min(H,y2+m)
    ring_mask = np.ones((ry2-ry1, rx2-rx1), bool)
    ring_mask[cy1-ry1:cy2-ry1, cx1-rx1:cx2-rx1] = False
    ring = img[ry1:ry2, rx1:rx2][ring_mask]
    if core.size == 0 or ring.size < 10: return None
    return float((core.mean()-ring.mean())/(ring.std()+1e-6))

# 去重 + 限流
by_seq = defaultdict(list)
for seq, frame, stem, b, c in raw:
    by_seq[seq].append((frame, stem, b, c))

pool = []
for seq, items in by_seq.items():
    items.sort(key=lambda x: -x[3])          # 分数降序
    kept = []                                 # (frame, stem, box, conf)
    for frame, stem, b, c in items:
        if any(iou(b, kb) > HOTSPOT_IOU for _,_,kb,_ in kept):
            continue                          # 同热点已有更高分
        if any(abs(frame - kf) < MIN_GAP for kf,_,_,_ in kept):
            continue                          # 时间间隔不足
        kept.append((frame, stem, b, c))
        if len(kept) >= CAP_PER_SEQ:
            break
    pool.extend([{'seq': seq, 'frame': f, 'stem': st, 'box': b, 'conf': c} for f, st, b, c in kept])

# 几何/对比度统计 + 可疑分档
img_cache = {}
suspicious = []
for it in pool:
    st = it['stem']
    if st not in img_cache:
        img_cache.clear()
        img_cache[st] = cv2.imread(str(TRAIN/'images'/f'{st}.jpg'), 0)
    img = img_cache[st]
    x1,y1,x2,y2 = it['box']
    w,h = x2-x1, y2-y1
    d = math.hypot(w,h); r = max(w,h)/max(1e-6,min(w,h))
    c = contrast(img, it['box']) if img is not None else None
    it.update({'d': round(d,1), 'ratio': round(r,2), 'contrast': round(c,2) if c is not None else None,
               'polarity': ('hot' if c and c > 0 else 'cold') if c is not None else '?'})
    # 可疑档：无人机样（细长 1.5-6）+ 高对比(|c|>1.5) + 尺寸合理(6-40px)
    if c is not None and 1.5 <= r <= 6 and abs(c) > 1.5 and 6 <= d <= 40:
        suspicious.append(it)
        crop = img[max(0,int(y1)-10):int(y2)+10, max(0,int(x1)-10):int(x2)+10]
        if crop.size:
            cv2.imwrite(str(OUT/'suspicious'/f"{st}_c{it['conf']:.2f}_d{d:.0f}_c{c:+.1f}.jpg"),
                        cv2.resize(crop, (max(64,crop.shape[1]*4), max(64,crop.shape[0]*4)), interpolation=cv2.INTER_NEAREST))

strong = [it for it in pool if it['conf'] > 0.25]
print(f'\n池化后误报候选: {len(pool)} (强 conf>0.25: {len(strong)})，覆盖 {len(by_seq)} 序列')
print(f'可疑档（疑似漏标）: {len(suspicious)}，crop 已导出 {OUT}/suspicious/')
import collections
print('对比度极性分布:', dict(collections.Counter(it['polarity'] for it in pool)))
print('尺寸 d 分布: p25/p50/p90 =', np.percentile([it['d'] for it in pool], [25,50,90]).round(1).tolist())
print('conf 分布: p50/p90/max =', np.percentile([it['conf'] for it in pool], [50,90]).round(3).tolist(), round(max(it['conf'] for it in pool), 3))

(OUT/'pool.json').write_text(json.dumps({'pool': pool, 'n_suspicious': len(suspicious),
    'config': {'CONF_MIN': CONF_MIN, 'CAP_PER_SEQ': CAP_PER_SEQ, 'MIN_GAP': MIN_GAP, 'HOTSPOT_IOU': HOTSPOT_IOU}}, indent=1))
print(f'\nsaved -> {OUT}/pool.json')
