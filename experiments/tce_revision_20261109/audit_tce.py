"""Frozen-protocol audit for TCE arms, identical matcher/groups machinery as
cst_gap_confirmation_20261019 (audit_validation.py via the completed-review
module). For T3 the validator filters predictions to the current-frame class
(class 2) before matching -- the deployable output; ghost-channel predictions
never count as detections or background false positives."""
import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY = ROOT / 'experiments/tce_revision_20261109'
OLD = ROOT / 'experiments/aat_temporal_100e_20261017/completed_review_20261017'
spec = importlib.util.spec_from_file_location('prior_eval', OLD / 'evaluate_cst_completed.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class CurrentClassCompletedValidator(prior.CompletedValidator):
    CURRENT_CLS = 2

    @staticmethod
    def _filter(p, cls_id):
        keep = p['cls'] == cls_id
        return {'bboxes': p['bboxes'][keep], 'conf': p['conf'][keep],
                'cls': p['cls'][keep], 'extra': p['extra'][keep]}

    def update_metrics(self, preds, batch):
        preds = [self._filter(p, self.CURRENT_CLS) for p in preds]
        super().update_metrics(preds, batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['t2', 't3', 't3v2'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--split', choices=['val', 'test'], required=True)
    a = ap.parse_args()
    name = f'TCE_{a.arm.upper()}_seed{a.seed}'
    index = json.loads((ROOT / 'data/yolo/attribute_index.json').read_text())[a.split]
    prior.INDEX = index
    prior.av.INDEX = index
    prior.REPEAT_CURRENT = False
    run_root = ROOT / 'runs/tce_revision_20261109'
    # The live run may live in a "-2" suffixed dir if an aborted earlier
    # attempt occupied the canonical name (Ultralytics auto-increment).
    run_dir = run_root / name
    if not (run_dir / 'weights/best.pt').is_file():
        run_dir = run_root / f'{name}-2'
    weights = run_dir / 'weights/best.pt'
    data = STUDY / f'cst_tce_{"t3" if a.arm == "t3v2" else a.arm}.yaml'
    dest = STUDY / 'audit' / a.split / name
    assert not dest.exists()
    assert weights.is_file()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(
        10 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    model = YOLO(str(weights))
    validator_cls = CurrentClassCompletedValidator if a.arm != 't2' else prior.CompletedValidator
    metrics = model.val(validator=validator_cls, data=str(data), split=a.split,
        imgsz=640, batch=128, workers=0, device='0', half=True, conf=.001, iou=.7,
        max_det=300, rect=True, plots=False, save_json=False,
        project=str(dest.parent), name=dest.name, exist_ok=False, verbose=False)
    assert metrics.audit_shapes == {'(3, 544, 672)': len(index)}, metrics.audit_shapes
    report = dict(name=name, arm=a.arm, seed=a.seed, split=a.split,
        weights_sha256=sha(weights), script_sha256=sha(__file__),
        completed_at=datetime.now(timezone.utc).isoformat(),
        protocol=dict(batch=128, imgsz=640, actual_shapes=metrics.audit_shapes, half=True,
            conf=.001, nms_iou=.7, max_det=300, rect=True,
            selection='native val AP50-95 best.pt, no test selection',
            current_class_filter=(a.arm != 't2')),
        metrics={k: float(v) for k, v in metrics.results_dict.items()},
        groups=json.loads((dest / 'groups.json').read_text()))
    (dest / 'audit.json').write_text(json.dumps(report, indent=2))
    print('AUDIT_COMPLETE', name, a.split, report['metrics'], flush=True)


if __name__ == '__main__':
    main()
