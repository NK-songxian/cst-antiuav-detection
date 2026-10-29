"""CPU-only, training-only translation diagnostic; never a camera-motion certificate."""
import json
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
OUT = ROOT / 'experiments/structure_ab_100e_20261017'
cv2.setNumThreads(1)
index = json.loads((ROOT / 'data/yolo/attribute_index.json').read_text())['train']
groups = defaultdict(list)
for text in (ROOT / 'experiments/baselines_100e_20261017/train.txt').read_text().splitlines():
    p = Path(text)
    groups[index[p.stem]['seq']].append(p)
records = []
for seq, paths in sorted(groups.items()):
    paths.sort(key=lambda p: index[p.stem]['frame'])
    lookup = {index[p.stem]['frame']: p for p in paths}
    for gap in (5, 10):
        candidates = [(p, lookup[index[p.stem]['frame'] - gap]) for p in paths
                      if index[p.stem]['frame'] - gap in lookup]
        if not candidates:
            continue
        current, previous = candidates[len(candidates) // 2]
        a = cv2.imread(str(previous), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        b = cv2.imread(str(current), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        h, w = a.shape
        mask = np.ones_like(a)
        for p in (current, previous):
            label = p.parent.parent / 'labels' / (p.stem + '.txt')
            if label.exists():
                for line in label.read_text().splitlines():
                    _, x, y, bw, bh = map(float, line.split())
                    x1, x2 = max(0, int((x-bw/2)*w)-12), min(w, int((x+bw/2)*w)+13)
                    y1, y2 = max(0, int((y-bh/2)*h)-12), min(h, int((y+bh/2)*h)+13)
                    mask[y1:y2, x1:x2] = 0
        a = (a - a[mask > 0].mean()) * mask
        b = (b - b[mask > 0].mean()) * mask
        window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(a, b, window)
        records.append(dict(seq=seq, gap=gap, current=current.name, previous=previous.name,
                            dx=dx, dy=dy, shift_px=float(np.hypot(dx, dy)), response=response))
summary = {}
for gap in (5, 10):
    rows = [r for r in records if r['gap'] == gap]
    summary[gap] = {'pairs': len(rows)}
    for cutoff in (0.0, 0.3, 0.6):
        selected = [r for r in rows if r['response'] >= cutoff]
        v = np.array([r['shift_px'] for r in selected])
        summary[gap][str(cutoff)] = dict(n=len(selected), median_px=float(np.median(v)),
            p90_px=float(np.quantile(v, .9)), fraction_gt_1px=float(np.mean(v > 1)),
            fraction_gt_2px=float(np.mean(v > 2))) if len(v) else {'n': 0}
result = {'protocol': 'one middle pair per training sequence per gap; GT boxes masked by union +12px; Hanning phase correlation; translation diagnostic only',
          'summary': summary, 'records': records}
(OUT / 'temporal_background_translation_audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps({'protocol': result['protocol'], 'summary': summary}, indent=2))
