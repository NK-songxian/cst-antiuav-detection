"""Re-evaluate completed checkpoints with the frozen baseline validator."""
import argparse
import hashlib
import json
import os
import sys
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
BASE = ROOT/'experiments/baselines_100e_20261017'
STUDY = ROOT/'experiments/structure_ab_100e_20261017'
RUNS = ROOT/'runs/structure_ab_100e_20261017'
sys.path.insert(0, str(BASE))
av = types.ModuleType('frozen_audit_for_completed')
frozen_source = (BASE/'audit_validation.py').read_text()
assert frozen_source.count('== 9066') == 1
# Only generalize the expected frame count; preserve the original AP and grouping implementation.
exec(compile(frozen_source.replace('== 9066', '== len(INDEX)'),
             str(BASE/'audit_validation.py'), 'exec'), av.__dict__)


class CompletedValidator(av.AuditValidator):
    def preprocess(self, batch):
        batch = super().preprocess(batch)
        if not hasattr(self, 'shapes'):
            self.shapes = Counter()
        self.shapes[str(tuple(batch['img'].shape[1:]))] += len(batch['img'])
        return batch

    def get_stats(self):
        result = super().get_stats()
        self.metrics.audit_shapes = dict(self.shapes)
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', choices=['A_zero_deploy','B_p2_compact','yolo11n','AAT_v1_legacy'], required=True)
    ap.add_argument('--square', action='store_true')
    ap.add_argument('--split',choices=['val','test'],default='val')
    args = ap.parse_args()
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    av.INDEX = json.loads((ROOT/'data/yolo/attribute_index.json').read_text())[args.split]
    if args.name=='yolo11n':
        weights = ROOT/'runs/baselines_100e_20261017/yolo11n/weights/best.pt'
    elif args.name=='AAT_v1_legacy':
        old=Path(json.loads((ROOT/'baseline_reset_latest.json').read_text())['old_runs'])
        weights=old/'AAT_v1_yolo11n/weights/best.pt'
    else:
        weights=RUNS/args.name/'weights/deploy.pt'
    protocol = 'square640' if args.square else 'baseline_rect'
    if args.split!='val':protocol += '_'+args.split
    dest = RUNS/'final_audit'/protocol/args.name
    assert not dest.exists(), dest
    model = YOLO(str(weights))
    start = datetime.now(timezone.utc).isoformat()
    metrics = model.val(validator=CompletedValidator, data=str(BASE/'cst_original.yaml'),
        split=args.split,imgsz=640,batch=128,workers=4,device='0',half=True,
        conf=0.001,iou=0.7,max_det=300,rect=not args.square,plots=False,save_json=False,
        project=str(dest.parent),name=dest.name,exist_ok=False,verbose=False)
    result = {'name':args.name,'weights':str(weights),'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
        'validator_source_sha256':hashlib.sha256((BASE/'audit_validation.py').read_bytes()).hexdigest(),
        'evaluation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'started_at':start,'ended_at':datetime.now(timezone.utc).isoformat(),
        'protocol':dict(split=args.split,frames=len(av.INDEX),includes_negatives=True,batch=128,workers=4,half=True,
             rect=not args.square,imgsz=640,conf=0.001,nms_iou=0.7,max_det=300,
             ap_implementation='Ultralytics DetMetrics',no_training=True,actual_shapes=metrics.audit_shapes),
        'metrics':{k:float(v) for k,v in metrics.results_dict.items()},
        'groups':json.loads((dest/'groups.json').read_text()),
        'speed_ms_per_image_batched':metrics.speed}
    (dest/'audit.json').write_text(json.dumps(result,indent=2))
    print('AUDIT_COMPLETE',json.dumps({k:result[k] for k in ['name','metrics','protocol']}),flush=True)


if __name__=='__main__':
    main()
