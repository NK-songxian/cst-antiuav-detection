"""Verify gap10 history semantics against the frozen aat reference, then scan
ghost-label completion statistics under candidate IoU rules to pin the frozen
definition that reproduces the paper's diagnostic (86.3% of stacked train
frames contain unlabeled residual positives, mean 1.97 boxes).

Ghost rule candidates (near channel preferred over far):
  near box is a ghost iff max IoU vs all current boxes < T.
  far  box is a ghost iff max IoU vs current < T AND vs kept near ghosts < T.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY = ROOT / 'experiments/tce_revision_20261109'
AAT = ROOT / 'experiments/aat_temporal_100e_20261017'


def build_history(index, near, far):
    groups = defaultdict(dict)
    for stem, rec in index.items():
        groups[rec['seq']][rec['frame']] = stem
    result = {}
    for stem, rec in index.items():
        lookup = groups[rec['seq']]
        frame = rec['frame']
        p_near = lookup.get(frame - near, stem)
        p_far = lookup.get(frame - far, p_near)
        result[stem] = [p_far, p_near, stem]
    return result


def boxes(stem, split):
    p = ROOT / 'data/yolo' / split / 'labels' / (stem + '.txt')
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            cx, cy, w, h = map(float, parts[1:5])
            out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area - inter + 1e-12) if area > 0 else 0.0


def ghosts(triplet, split, thresh):
    far_s, near_s, cur_s = triplet
    cur = boxes(cur_s, split)
    kept = []
    near_dup = 0
    for b in boxes(near_s, split):
        if all(iou(b, c) < thresh for c in cur):
            if all(iou(b, g) < thresh for g in kept):
                kept.append(b)
            else:
                near_dup += 1
    far_ghost = 0
    for b in boxes(far_s, split):
        if all(iou(b, c) < thresh for c in cur) and all(iou(b, g) < thresh for g in kept):
            far_ghost += 1
    return len(kept), far_ghost


def main():
    index = json.loads((ROOT / 'data/yolo/attribute_index.json').read_text())
    ok = True
    for split in ('train', 'val'):
        mine = build_history(index[split], near=5, far=10)
        ref = json.loads((AAT / f'history_{split}.json').read_text())
        same = mine == ref
        ok &= same
        print(f'[{split}] build_history==aat reference: {same}')
        if not same:
            diff = [k for k in ref if mine.get(k) != ref[k]][:5]
            print('  sample mismatches:', diff)
    assert ok, 'history semantics mismatch'

    train = index['train']
    history = json.loads((AAT / 'history_train.json').read_text())
    n_missing_labels = sum(1 for s in train
                           if not (ROOT / 'data/yolo/train/labels' / (s + '.txt')).exists())
    print(f'train stems={len(train)} missing label files={n_missing_labels}')

    for thresh in (0.5, 0.1, 1e-9):
        frames_with = 0
        total_ghost = 0
        ghost_only_sum = 0
        n_exist1_ghost = 0
        exist1 = 0
        for stem, rec in train.items():
            n_near, n_far = ghosts(history[stem], 'train', thresh)
            g = n_near + n_far
            total_ghost += g
            if g:
                frames_with += 1
                ghost_only_sum += g
                if rec['exist'] == 1:
                    n_exist1_ghost += 1
            if rec['exist'] == 1:
                exist1 += 1
        n = len(train)
        print(f'T={thresh}: frames_with_ghost={frames_with / n:.3f} '
              f'mean_ghost_all={total_ghost / n:.3f} '
              f'mean_ghost_if_any={ghost_only_sum / max(1, frames_with):.3f} '
              f'exist1_with_ghost={n_exist1_ghost / max(1, exist1):.3f}')


if __name__ == '__main__':
    main()
