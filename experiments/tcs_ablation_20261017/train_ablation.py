"""TCS ablation runner: train one job (100e, patience 20, batch 256 — same as C1).

Jobs: tcs_yolo8n / tcs_yolo11s reuse the aat_temporal materialized data (delta=5);
delta1 / delta10 use locally materialized datasets. SharedBrightness keeps one
HSV LUT across time channels (same as C1).
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY = ROOT / 'experiments/tcs_ablation_20261017'
AAT = ROOT / 'experiments/aat_temporal_100e_20261017'

JOBS = {
    'tcs_yolo8n': dict(weights=ROOT / 'yolov8n.pt', data=AAT / 'cst_temporal.yaml'),
    'tcs_yolo11s': dict(weights=ROOT / 'yolo11s.pt', data=AAT / 'cst_temporal.yaml'),
    'tcs_far15_yolo11n': dict(weights=ROOT / 'yolo11n.pt', data=STUDY / 'cst_far15.yaml'),
    'tcs_far20_yolo11n': dict(weights=ROOT / 'yolo11n.pt', data=STUDY / 'cst_far20.yaml'),
}


class SharedBrightness:
    """Same V gain draw as RandomHSV; one LUT across time, no time-channel mixing."""

    def __init__(self, hgain=0.015, sgain=0.7, vgain=0.4):
        self.vgain = vgain

    def __call__(self, labels):
        import cv2
        r = np.random.uniform(-1, 1, 3)
        lut = np.clip(np.arange(256, dtype=np.float64) * (1 + r[2] * self.vgain), 0, 255).astype(np.uint8)
        labels['img'] = cv2.LUT(labels['img'], lut)
        return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, choices=sorted(JOBS))
    args = ap.parse_args()
    job = JOBS[args.name]

    import ultralytics.data.augment as augment
    augment.RandomHSV = SharedBrightness

    run = STUDY / 'runs' / args.name
    assert not run.exists(), f'{run} already exists'
    model = YOLO(str(job['weights']))
    model.train(
        data=str(job['data']), project=str(STUDY / 'runs'), name=args.name,
        epochs=100, patience=20, batch=256, imgsz=640, workers=16,
        seed=0, device=0, amp=True, optimizer='auto', cos_lr=False,
        close_mosaic=10, rect=False, single_cls=False, cache=False,
        deterministic=True, exist_ok=False, verbose=True,
    )
    csv = run / 'results.csv'
    best = None
    if csv.exists():
        import csv as csvmod
        rows = list(csvmod.DictReader(csv.open()))
        if rows:
            b = max(rows, key=lambda r: float(r['metrics/mAP50(B)']))
            best = dict(epoch=b['epoch'], mAP50=b['metrics/mAP50(B)'],
                        mAP50_95=b['metrics/mAP50-95(B)'])
    (STUDY / f'{args.name}.result.json').write_text(json.dumps(dict(
        name=args.name, finished_at=datetime.now(timezone.utc).isoformat(), best=best), indent=2))
    print(f'JOB_DONE {args.name} best={best}')


if __name__ == '__main__':
    main()
