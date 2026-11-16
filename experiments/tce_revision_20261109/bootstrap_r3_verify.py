# -*- coding: utf-8 -*-
"""实跑事件级配对 bootstrap，验证 p59/p61 的两个 95% 区间。"""
import json, struct, numpy as np
from pathlib import Path

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
GAP = ROOT / 'experiments/cst_gap_confirmation_20261019/analysis'
TCE = ROOT / 'experiments/tce_revision_20261109'

def png_size(p):
    with open(p, 'rb') as f:
        f.seek(16)
        w, h = struct.unpack('>II', f.read(8))
    return w, h

def iou_matrix(a, b):
    # a: N×4, b: M×4 (xyxy)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return np.where(ua > 0, inter / ua, 0.0)

events = json.loads((GAP / 'reappearance_analysis.json').read_text())['events']
ev_seq = [e['seq'] for e in events]
seqs = sorted(set(ev_seq))
assert len(seqs) == 47 and len(events) == 178

# ---------- 单帧 / 三帧 per-event hits（存档数据） ----------
hit_sf = {}   # hit_sf[('单帧', seed)] = 178×3 bool
hit_tf = {}
ra = json.loads((GAP / 'reappearance_analysis.json').read_text())['models']
for seed in range(3):
    d = {f"{m['seq']}|{m['start_frame']}": m['hits'] for m in ra[f'单帧seed{seed}']['details']}
    hit_sf[seed] = [d[f"{s}|{e['start_frame']}"] for s, e in zip(ev_seq, events)]
    d = {f"{m['seq']}|{m['start_frame']}": m['hits'] for m in ra[f'时序seed{seed}']['details']}
    hit_tf[seed] = [d[f"{s}|{e['start_frame']}"] for s, e in zip(ev_seq, events)]

# ---------- TCE per-event hits ----------
tau_tce = {0: 0.421, 1: 0.437, 2: 0.491}
hit_tce = {}
event_frames = sorted({st for e in events for st in e['stems']})
sizes = {st: png_size(TCE / f'data/t3/test/images/{st}.png') for st in event_frames}

def r3(hits):   # hits: 178×3
    return np.mean([any(h) for h in hits]) * 100

for seed in range(3):
    preds = json.loads((TCE / f'audit/test/TCE_T3V2_seed{seed}/predictions_conf010.json').read_text())
    tau = tau_tce[seed]
    ev_hit = []
    for s, e in zip(ev_seq, events):
        ok3 = [False, False, False]
        for k, st in enumerate(e['stems']):
            lab = TCE / f'data/t3/test/labels/{st}.txt'
            W, H = sizes[st]
            gt = []
            for line in lab.read_text().splitlines():
                c, x, y, w, h = line.split()
                if c == '2':
                    x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
                    gt.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            pr = preds.get(st)
            if pr and gt:
                conf = np.array(pr['conf']); m = conf >= tau
                if m.any():
                    M = iou_matrix(np.array(pr['xyxy'])[m], np.array(gt))
                    ok3[k] = bool((M >= 0.5).any())
        ev_hit.append(ok3)
    hit_tce[seed] = ev_hit
    r1 = np.mean([h[0] for h in ev_hit]) * 100
    print(f'TCE seed{seed}: R1={r1:.2f} R3={r3(ev_hit):.2f}  (记录: R1 30.90±0.97, R3 49.63±0.32)')

# 回归验证：全测试集 conf0.25 正帧 hit rate（seed0）
preds = json.loads((TCE / 'audit/test/TCE_T3V2_seed0/predictions_conf010.json').read_text())
audit = json.loads((TCE / 'audit/test/TCE_T3V2_seed0/audit.json').read_text())
ref = audit['groups']['ALL']['conf_0.25']['positive_frame_hit_rate_iou50']
all_stems = [l.strip().split('/')[-1][:-4] for l in (TCE / 't3_test.txt').read_text().splitlines() if l.strip()]
n_hit = n_pos = 0
for st in all_stems:
    lab = TCE / f'data/t3/test/labels/{st}.txt'
    W, H = sizes.get(st, png_size(TCE / f'data/t3/test/images/{st}.png'))
    gt = []
    for line in lab.read_text().splitlines():
        c, x, y, w, h = line.split()
        if c == '2':
            x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
            gt.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
    if not gt:
        continue
    n_pos += 1
    pr = preds.get(st)
    if pr:
        conf = np.array(pr['conf']); m = conf >= 0.25
        if m.any() and (iou_matrix(np.array(pr['xyxy'])[m], np.array(gt)) >= 0.5).any():
            n_hit += 1
print(f'回归: 全测试集正帧hit {n_hit}/{n_pos} = {n_hit/n_pos:.6f}  audit基准 {ref:.6f}  diff={abs(n_hit/n_pos-ref)*100:.4f}pp')

# ---------- bootstrap：47 序列有放回 × 20000，3种子均值差 ----------
seq_events = [[i for i, s in enumerate(ev_seq) if s == q] for q in seqs]
rng = np.random.default_rng(20261114)
N = 20000

def r3_boot(hits_by_seed, idx):
    # hits_by_seed: {seed: 178 hits list}; idx: 选中的事件索引（序列加权）
    out = []
    for seed in range(3):
        hh = [hits_by_seed[seed][i] for i in idx]
        out.append(r3(hh))
    return np.mean(out)

sf_r3 = [r3(hit_sf[s]) for s in range(3)]
tf_r3 = [r3(hit_tf[s]) for s in range(3)]
tce_r3 = [r3(hit_tce[s]) for s in range(3)]
print(f'逐种子 R3: 单帧 {[round(v,2) for v in sf_r3]}  三帧 {[round(v,2) for v in tf_r3]}  TCE {[round(v,2) for v in tce_r3]}')
print(f'均值差: TCE-单帧 = {np.mean(tce_r3)-np.mean(sf_r3):.2f}  TCE-三帧 = {np.mean(tce_r3)-np.mean(tf_r3):.2f}')

boot_sf, boot_tf = [], []
for _ in range(N):
    pick = rng.choice(len(seqs), len(seqs), replace=True)
    idx = [i for pi in pick for i in seq_events[pi]]
    boot_sf.append(r3_boot(hit_tce, idx) - r3_boot(hit_sf, idx))
    boot_tf.append(r3_boot(hit_tce, idx) - r3_boot(hit_tf, idx))

ci_sf = np.percentile(boot_sf, [2.5, 97.5])
ci_tf = np.percentile(boot_tf, [2.5, 97.5])
print(f'TCE-单帧 95%CI = [{ci_sf[0]:.2f}, {ci_sf[1]:.2f}]   论文 p59: [10.11, 19.67]')
print(f'TCE-三帧 95%CI = [{ci_tf[0]:.2f}, {ci_tf[1]:.2f}]   论文 p61: [-4.61, 0.21]')
