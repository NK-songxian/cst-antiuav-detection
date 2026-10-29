"""Read-only, train-only feasibility probe. Does not train or modify source data."""
import argparse
import bisect
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    index = json.loads((root / 'data/yolo/attribute_index.json').read_text())['train']
    seqs = defaultdict(list)
    for stem, rec in index.items():
        if rec.get('exist') == 1:
            seqs[rec['seq']].append((rec['frame'], stem, rec))
    for rows in seqs.values():
        rows.sort()
    selected = sorted(seqs, key=lambda s: sum(r[2]['attr'].get('TC') == 1 for r in seqs[s]), reverse=True)[:8]
    cache = {}

    def sample(stem):
        if stem in cache:
            return cache[stem]
        image = cv2.imread(str(root / 'data/yolo/train/images' / (stem + '.jpg')), 0)
        label = root / 'data/yolo/train/labels' / (stem + '.txt')
        if image is None or not label.exists() or not label.read_text().strip():
            cache[stem] = None
            return None
        row = list(map(float, label.read_text().splitlines()[0].split()))
        height, width = image.shape
        cx, cy, bw, bh = np.array(row[1:5]) * [width, height, width, height]
        x1, y1 = int(np.floor(cx-bw/2)), int(np.floor(cy-bh/2))
        x2, y2 = int(np.ceil(cx+bw/2)), int(np.ceil(cy+bh/2))
        x1, y1, x2, y2 = max(0,x1),max(0,y1),min(width,x2),min(height,y2)
        if x2-x1 < 3 or y2-y1 < 3:
            cache[stem] = None
            return None
        margin = max(4, int(max(bw,bh)/2))
        a,b,c,d = max(0,x1-margin),max(0,y1-margin),min(width,x2+margin),min(height,y2+margin)
        context = image[b:d,a:c].astype(float)
        mask = np.ones(context.shape, dtype=bool)
        mask[y1-b:y2-b,x1-a:x2-a] = False
        ring = context[mask]
        patch = image[y1:y2,x1:x2].astype(float)
        if ring.size < 8:
            cache[stem] = None
            return None
        contrast = abs(float(patch.mean())-float(ring.mean())) / max(float(ring.std()),1.0)
        result = dict(contrast=contrast, area=float(bw*bh), box=[x1,y1,x2,y2])
        cache[stem] = result
        return result

    pairs = []
    attempted = 0
    for seq in selected:
        rows = seqs[seq]
        tc = [r for r in rows if r[2]['attr'].get('TC') == 1]
        if not tc:
            continue
        frames = [r[0] for r in rows]
        for pos in np.unique(np.linspace(0,len(tc)-1,min(10,len(tc)),dtype=int)):
            frame, stem, rec = tc[int(pos)]
            attempted += 1
            current = sample(stem)
            if current is None:
                continue
            candidates=[]
            for other_frame, other_stem, _ in rows[bisect.bisect_left(frames,frame-50):bisect.bisect_right(frames,frame+50)]:
                if other_stem == stem:
                    continue
                candidate=sample(other_stem)
                if candidate and .5 <= candidate['area']/current['area'] <= 2:
                    candidates.append((candidate['contrast'],other_frame,other_stem,candidate))
            if not candidates:
                continue
            _, other_frame, other_stem, candidate=max(candidates)
            pairs.append(dict(seq=seq,hard=stem,reference=other_stem,gap_frames=abs(frame-other_frame),hard_proxy=current['contrast'],reference_proxy=candidate['contrast'],hard_box=current['box'],reference_box=candidate['box']))

    # Check whether the same detector is actually more reliable on the selected reference.
    # The radiometric proxy above is a proposal heuristic, not a ground-truth clarity label.
    from ultralytics import YOLO
    detector=YOLO(str(root/'runs/baseline_yolo11n/weights/best.pt'))
    stems=sorted({p[k] for p in pairs for k in ['hard','reference']})
    pred={}
    for start in range(0,len(stems),16):
        batch=stems[start:start+16]
        for stem,result in zip(batch,detector.predict([str(root/'data/yolo/train/images'/(s+'.jpg')) for s in batch],imgsz=640,device=0,conf=.001,verbose=False,stream=True)):
            gt=np.array(sample(stem)['box'],dtype=float)
            boxes=result.boxes.xyxy.cpu().numpy()
            scores=result.boxes.conf.cpu().numpy()
            inter=np.clip(np.minimum(boxes[:,2:],gt[2:])-np.maximum(boxes[:,:2],gt[:2]),0,None).prod(1)
            union=np.clip(boxes[:,2:]-boxes[:,:2],0,None).prod(1)+(gt[2]-gt[0])*(gt[3]-gt[1])-inter
            ious=inter/np.maximum(union,1e-9)
            valid=scores[ious>=.5]
            pred[stem]=float(valid.max()) if valid.size else 0.
    for pair in pairs:
        pair['hard_matched_conf']=pred[pair['hard']]
        pair['reference_matched_conf']=pred[pair['reference']]
        pair['reference_qualified']=pred[pair['reference']]>=.25 and pred[pair['reference']]>=pred[pair['hard']]+.1
    report=dict(scope='training split only; exploratory, eight TC-rich sequences, at most ten sampled frames each',attempted=attempted,pairs_found=len(pairs),qualified=sum(p['reference_qualified'] for p in pairs),qualified_by_sequence=dict(Counter(p['seq'] for p in pairs if p['reference_qualified'])),median_proxy_gain=float(np.median([p['reference_proxy']-p['hard_proxy'] for p in pairs])) if pairs else None,pairs=pairs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='pairs'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
