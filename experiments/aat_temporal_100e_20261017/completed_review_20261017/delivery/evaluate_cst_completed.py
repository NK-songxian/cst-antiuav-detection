"""Fixed-checkpoint CST evaluation; preserve the existing AP matcher and all negatives."""
import argparse
import hashlib
import json
import os
import sys
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data import build

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
BASE = ROOT/'experiments/baselines_100e_20261017'
STUDY = ROOT/'experiments/aat_temporal_100e_20261017'
OUT = STUDY/'completed_review_20261017'
NAMES = ['yolov8n', 'yolo11n', 'yolov5nu', 'yolo26n', 'A1_aat_global',
         'A2_aat_sequence', 'C1_temporal_uniform', 'C2_temporal_aat_sequence']
INDEX = None
REPEAT_CURRENT = False
av = types.ModuleType('cst_completed_frozen_audit')
source = (BASE/'audit_validation.py').read_text()
assert source.count('== 9066') == 1
exec(compile(source.replace('== 9066', '== len(INDEX)'), str(BASE/'audit_validation.py'), 'exec'), av.__dict__)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CompletedValidator(av.AuditValidator):
    def preprocess(self, batch):
        batch = super().preprocess(batch)
        if REPEAT_CURRENT:
            # Network RGB order is [current, t-5, t-10]. Only a diagnostic,
            # evaluated with the same checkpoint; not a trained single-frame baseline.
            batch['img'][:, 1:] = batch['img'][:, :1].clone()
        if not hasattr(self, 'shape_counts'):
            self.shape_counts = Counter()
        self.shape_counts[str(tuple(batch['img'].shape[1:]))] += len(batch['img'])
        return batch

    def update_metrics(self, preds, batch):
        if not hasattr(self, 'saved_boxes'):
            self.saved_boxes = {}
        super().update_metrics(preds, batch)
        for i, pred in enumerate(preds):
            pbatch = self._prepare_batch(i, batch)
            scaled = self.scale_preds(pred, pbatch)
            keep = scaled['conf'] >= 0.1
            self.saved_boxes[Path(batch['im_file'][i]).stem] = {
                'xyxy': scaled['bboxes'][keep].float().cpu().tolist(),
                'conf': scaled['conf'][keep].float().cpu().tolist()}

    def get_stats(self):
        result = super().get_stats()
        self.metrics.audit_shapes = dict(self.shape_counts)
        (self.save_dir/'predictions_conf010.json').write_text(json.dumps(self.saved_boxes))
        return result


def main():
    global INDEX, REPEAT_CURRENT
    p = argparse.ArgumentParser()
    p.add_argument('--name', choices=NAMES, required=True)
    p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--repeat-current', action='store_true')
    args = p.parse_args()
    REPEAT_CURRENT = args.repeat_current
    temporal = args.name.startswith('C')
    assert not REPEAT_CURRENT or temporal
    index_path = ROOT/'data/yolo/attribute_index.json'
    INDEX = json.loads(index_path.read_text())[args.split]
    av.INDEX = INDEX
    model_study = 'baselines_100e_20261017' if args.name in NAMES[:4] else 'aat_temporal_100e_20261017'
    weights = ROOT/'runs'/model_study/args.name/'weights/best.pt'
    assert weights.is_file()
    data = OUT/'cst_temporal_evaluation.yaml' if temporal else BASE/'cst_original.yaml'
    dest = OUT/'evaluations'/args.split/(args.name+('_repeat_current' if REPEAT_CURRENT else ''))
    assert not dest.exists(), dest
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(0).total_memory)
    original_loader = build.InfiniteDataLoader

    class BoundedLoader(original_loader):
        def __init__(self, *pos, **kw):
            if kw.get('num_workers', 0):
                kw.update(prefetch_factor=1, multiprocessing_context='forkserver')
            kw['pin_memory'] = False
            super().__init__(*pos, **kw)

    build.InfiniteDataLoader = BoundedLoader
    started = datetime.now(timezone.utc).isoformat()
    model = YOLO(str(weights))
    parameters_before_fusion = sum(p.numel() for p in model.model.parameters())
    metrics = model.val(validator=CompletedValidator, data=str(data), split=args.split,
        imgsz=640, batch=128, workers=2, device='0', half=True, conf=0.001,
        iou=0.7, max_det=300, rect=True, plots=False, save_json=False,
        project=str(dest.parent), name=dest.name, exist_ok=False, verbose=False)
    report = dict(name=args.name, diagnostic_repeat_current=REPEAT_CURRENT,
        weights=str(weights), weights_sha256=sha(weights), parameters_before_fusion=parameters_before_fusion,
        validator_source_sha256=sha(BASE/'audit_validation.py'), evaluation_source_sha256=sha(__file__),
        index_sha256=sha(index_path), started_at=started, ended_at=datetime.now(timezone.utc).isoformat(),
        protocol=dict(split=args.split, frames=len(INDEX), includes_negatives=True,
            batch=128, workers=2, half=True, rect=True, imgsz=640, conf=0.001,
            nms_iou=0.7, max_det=300, temporal=temporal,
            checkpoint_selection='native best.pt selected on validation AP50-95; no test selection',
            actual_shapes=metrics.audit_shapes, ap_implementation='Ultralytics DetMetrics'),
        metrics={k: float(v) for k, v in metrics.results_dict.items()},
        groups=json.loads((dest/'groups.json').read_text()),
        speed_ms_per_image_batched=metrics.speed,
        speed_warning='Shared GPU; batched timings are diagnostic only, not deployment latency.')
    (dest/'audit.json').write_text(json.dumps(report, indent=2))
    print('AUDIT_COMPLETE', json.dumps({k: report[k] for k in ['name', 'diagnostic_repeat_current', 'metrics']}), flush=True)


if __name__ == '__main__':
    main()
