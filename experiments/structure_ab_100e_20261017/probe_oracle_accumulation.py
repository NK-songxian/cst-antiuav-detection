"""Training-only image diagnostic using GT alignment, NOT detection performance."""
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
OUT = ROOT / 'experiments/structure_ab_100e_20261017'
cv2.setNumThreads(1)
idx = json.loads((ROOT/'data/yolo/attribute_index.json').read_text())['train']
paths = {Path(p).stem: Path(p) for p in (ROOT/'experiments/baselines_100e_20261017/train.txt').read_text().splitlines()}
by_frame = {(idx[k]['seq'], idx[k]['frame']): k for k in paths}

def box(k):
    p = paths[k]
    t = (p.parent.parent/'labels'/(p.stem+'.txt')).read_text().splitlines()
    return np.array(list(map(float, t[0].split()[1:]))) if t else None

candidates = {0: [], 1: []}
for k in sorted(paths):
    m = idx[k]
    keys = [by_frame.get((m['seq'], m['frame']-d)) for d in (10,5,0)]
    if any(v is None for v in keys):
        continue
    boxes = [box(v) for v in keys]
    if any(b is None for b in boxes):
        continue
    sizes = np.array([b[2:]*[640,512] for b in boxes])
    if np.any(sizes.max(0)/np.maximum(sizes.min(0), 1e-8) > 1.5):
        continue
    candidates[int(bool(m['attr']['TC']))].append((keys, boxes))

def contrast(a, bw, bh):
    yy, xx = np.mgrid[:65,:65]
    inner = (abs(xx-32) <= max(1,bw/2)) & (abs(yy-32) <= max(1,bh/2))
    outer = (abs(xx-32) <= max(6,2*bw)) & (abs(yy-32) <= max(6,2*bh))
    excluded = (abs(xx-32) <= max(3,bw)) & (abs(yy-32) <= max(3,bh))
    ring = a[outer & ~excluded]
    return float(abs(a[inner].mean()-ring.mean())/(ring.std()+1e-6))

records=[]
for tc, rows in candidates.items():
    for i in np.linspace(0,len(rows)-1,min(150,len(rows)),dtype=int):
        keys, boxes = rows[i]
        images = [cv2.imread(str(paths[k]),cv2.IMREAD_GRAYSCALE).astype(np.float32) for k in keys]
        # Current-frame center is fixed for unaligned averaging; GT center is used ONLY in oracle.
        cur = tuple((boxes[-1][:2]*[640,512]).tolist())
        aligned = [cv2.getRectSubPix(a,(65,65),tuple((b[:2]*[640,512]).tolist())) for a,b in zip(images,boxes)]
        unaligned = [cv2.getRectSubPix(a,(65,65),cur) for a in images]
        bw,bh = boxes[-1][2:]*[640,512]
        single=contrast(aligned[-1],bw,bh)
        oracle=contrast(np.mean(aligned,axis=0),bw,bh)
        plain=contrast(np.mean(unaligned,axis=0),bw,bh)
        records.append(dict(frame=keys[-1],tc=tc,single_cnr=single,unaligned_cnr=plain,
                            oracle_cnr=oracle,oracle_minus_single=oracle-single,
                            oracle_minus_unaligned=oracle-plain))
summary={}
for name, chosen in [('ALL',records),('TC',[r for r in records if r['tc']]),('non_TC',[r for r in records if not r['tc']])]:
    summary[name]={'n':len(chosen)}
    for key in ['single_cnr','unaligned_cnr','oracle_cnr','oracle_minus_single','oracle_minus_unaligned']:
        summary[name]['median_'+key]=float(np.median([r[key] for r in chosen]))
    summary[name]['fraction_oracle_gt_single']=float(np.mean([r['oracle_cnr']>r['single_cnr'] for r in chosen]))
result={'protocol':'Exploratory TRAIN-only GT-aligned image patches; deterministic evenly spaced 150 TC/150 non-TC; stable box size ratio <=1.5; gap5; CNR is abs(box mean-ring mean)/ring std; sampled frames correlated; not mAP, not detector upper bound, not an achievable deployment result',
        'summary':summary,'records':records}
(OUT/'oracle_accumulation_probe.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'protocol':result['protocol'],'summary':summary},indent=2))
