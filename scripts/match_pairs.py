#!/usr/bin/env python3
"""池分层 + 真假匹配（第2步）
①分层规则（依据核查结论.md）：
  suspect: 贴边(≤3px) OR 距同序列 exist 变化帧 ≤25 原始帧 OR 核查的可疑crop
           → 不做负监督，仅记录
  reliable: 其余 → 进入匹配
②匹配：对 reliable FP，从 train exist=1 真目标中按
  D = |log(d_p/d_n)| + 0.5|log(r_p/r_n)| + |log((|c_p|+e)/(|c_n|+e))|
  筛 尺度比0.5-2、对比比0.5-2、同极性优先，取前5近邻固定种子选1
输出: runs/fp_pool/pool_tiered.json, pairs.json, 覆盖率报告
"""
import json, math, random
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
TRAIN = ROOT/'data/yolo/train'
IDX = json.loads((ROOT/'data/yolo/attribute_index.json').read_text())['train']
POOL = json.loads((ROOT/'runs/fp_pool/pool.json').read_text())['pool']
OUT = ROOT/'runs/fp_pool'

# ---- 序列 exist 变化帧索引 ----
seq_frames = defaultdict(dict)
for stem, rec in IDX.items():
    seq_frames[rec['seq']][rec['frame']] = rec['exist']
exist_changes = {seq: sorted(f for f, e in fd.items()
                 for e2 in [fd.get(f-5, e)] if e != e2) for seq, fd in seq_frames.items()}
# 更简单：对每个序列记录所有相邻采样帧 exist 翻转点
change_pts = {}
for seq, fd in seq_frames.items():
    fs = sorted(fd)
    pts = set()
    for a, b in zip(fs, fs[1:]):
        if fd[a] != fd[b]:
            pts.add(b)
    change_pts[seq] = sorted(pts)

def box_stats(img, b):
    x1,y1,x2,y2 = b
    w,h = x2-x1, y2-y1
    d = math.hypot(w,h); r = max(w,h)/max(1e-6, min(w,h))
    H,W = img.shape
    cx1,cy1,cx2,cy2 = max(0,int(x1)),max(0,int(y1)),min(W,int(x2)),min(H,int(y2))
    core = img[cy1:cy2, cx1:cx2]
    m = max(2, int(round(max(w,h))))
    rx1,ry1,rx2,ry2 = max(0,int(x1)-m),max(0,int(y1)-m),min(W,int(x2)+m),min(H,int(y2)+m)
    ring_mask = np.ones((ry2-ry1, rx2-rx1), bool)
    ring_mask[cy1-ry1:cy2-ry1, cx1-rx1:cx2-rx1] = False
    ring = img[ry1:ry2, rx1:rx2][ring_mask]
    if core.size == 0 or ring.size < 10: return d, r, None
    c = float((core.mean()-ring.mean())/(ring.std()+1e-6))
    return d, r, c

def edge_hug(b, W=640, H=512, tol=3):
    return b[0] <= tol or b[1] <= tol or b[2] >= W-tol or b[3] >= H-tol

def near_change(seq, frame, win=25):
    return any(abs(frame - p) <= win for p in change_pts.get(seq, []))

# ---- ① 分层 ----
tiered = {'reliable': [], 'suspect': []}
for it in POOL:
    reasons = []
    if edge_hug(it['box']): reasons.append('edge')
    if near_change(it['seq'], it['frame']): reasons.append('near_exist_change')
    if it['contrast'] is None: reasons.append('no_contrast')
    it['tier_reasons'] = reasons
    (tiered['suspect'] if reasons else tiered['reliable']).append(it)
print(f"分层: reliable={len(tiered['reliable'])}  suspect={len(tiered['suspect'])} "
      f"(edge={sum('edge' in it['tier_reasons'] for it in tiered['suspect'])}, "
      f"near_change={sum('near_exist_change' in it['tier_reasons'] for it in tiered['suspect'])})", flush=True)

# ---- ② 真目标统计 ----
pos_stats = []   # (stem, box, d, r, c)
pos_frames = sorted(stem for stem, rec in IDX.items() if rec['exist'] == 1)
for i, stem in enumerate(pos_frames):
    img = cv2.imread(str(TRAIN/'images'/f'{stem}.jpg'), 0)
    if img is None: continue
    lbl = TRAIN/'labels'/f'{stem}.txt'
    if not lbl.exists(): continue
    for l in lbl.read_text().splitlines():
        if not l.strip(): continue
        p = l.split()
        cx,cy,w,h = (float(v)*s for v,s in zip(p[1:5], (640,512,640,512)))
        b = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
        d, r, c = box_stats(img, b)
        if c is not None:
            pos_stats.append({'stem': stem, 'box': b, 'd': d, 'r': r, 'c': c})
    if i % 5000 == 0:
        print(f'  目标统计 {i}/{len(pos_frames)}', flush=True)
print(f'真目标统计: {len(pos_stats)} 个（来自 {len(pos_frames)} 帧）', flush=True)

# ---- ③ 匹配 ----
rng = random.Random(2026)
pairs, unmatched = [], []
for it in tiered['reliable']:
    if it['contrast'] is None:
        unmatched.append(it); continue
    cn = abs(it['contrast']) + 1e-6
    cand = []
    for ps in pos_stats:
        if not (0.5 <= ps['d']/it['d'] <= 2.0): continue
        if not (0.5 <= abs(ps['c'])+1e-6/ (cn) <= 2.0) and not (0.5 <= cn/(abs(ps['c'])+1e-6) <= 2.0): continue
        D = (abs(math.log(ps['d']/it['d'])) + 0.5*abs(math.log(ps['r']/it['ratio']))
             + abs(math.log((abs(ps['c'])+1e-6)/cn)))
        cand.append((D, ps))
    cand.sort(key=lambda x: x[0])
    if not cand:
        unmatched.append(it); continue
    top = cand[:5]
    D, ps = rng.choice(top)
    pairs.append({'fp': it, 'pos': {'stem': ps['stem'], 'box': ps['box'], 'd': round(ps['d'],1),
                  'r': round(ps['r'],2), 'c': round(ps['c'],2)}, 'D': round(D,3)})

print(f'\n匹配结果: 配对成功 {len(pairs)}/{len(tiered["reliable"])} (覆盖率 {len(pairs)/max(1,len(tiered["reliable"]))*100:.0f}%)  未匹配 {len(unmatched)}')
if pairs:
    print('D 分布 p50/p90:', np.percentile([p['D'] for p in pairs], [50,90]).round(2).tolist())
    prot_stems = set(p['pos']['stem'] for p in pairs)
    print(f'被保护真目标: {len(prot_stems)} 帧（去重）')

(OUT/'pool_tiered.json').write_text(json.dumps(tiered, indent=1))
(OUT/'pairs.json').write_text(json.dumps({'pairs': pairs, 'unmatched': unmatched}, indent=1))
print(f'\nsaved -> {OUT}/pool_tiered.json, pairs.json')
