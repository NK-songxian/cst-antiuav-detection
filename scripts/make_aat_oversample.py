#!/usr/bin/env python3
"""AAT-v1: 属性先验离线重采样（attribute-prior oversampling）
按 attribute_index.json 的帧级属性，对 train 集困难帧做符号链接复制：
  TC/OC 困难帧 -> ×2（额外复制 1 份）    [遮挡/热交叉，最难]
  DBC/SIZE     -> ×1.5（50% 概率复制）  [低对比/极小目标]
val/test 不动。同时统计输出重采样分布。
"""
import json, random
from pathlib import Path
from collections import Counter

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav/data/yolo')
IDX = json.loads((ROOT / 'attribute_index.json').read_text())['train']
train_img = ROOT / 'train' / 'images'
train_lbl = ROOT / 'train' / 'labels'
n_before = len(list(train_img.glob('*.jpg')))

# 属性 -> (额外复制份数, 概率)   AAT-v2: 仅 TC（消融发现 SIZE/DBC 复制有害）
RULE = {'TC': (1, 1.0)}
rng = random.Random(2026)

stats = Counter()
for stem, rec in IDX.items():
    if rec.get('exist') != 1:
        continue
    src_i = train_img / f'{stem}.jpg'
    src_l = train_lbl / f'{stem}.txt'
    if not src_i.exists() or not src_l.exists():
        continue
    for attr, (copies, prob) in RULE.items():
        if not rec['attr'].get(attr):
            continue
        for c in range(copies):
            if rng.random() > prob:
                continue
            dst_i = train_img / f'{stem}__aat{c}.jpg'
            dst_l = train_lbl / f'{stem}__aat{c}.txt'
            if dst_i.is_symlink():
                continue
            dst_i.symlink_to(src_i.resolve())
            dst_l.symlink_to(src_l.resolve())
            stats[attr] += 1

n_after = len(list(train_img.glob('*.jpg')))
print(f'train images: {n_before} -> {n_after} (+{n_after-n_before})')
print('oversample by attr:', dict(stats))
for cache in (ROOT/'train'/'labels.cache', ROOT/'train'/'labels.cache.npy'):
    cache.unlink(missing_ok=True)
print('labels.cache cleared')
