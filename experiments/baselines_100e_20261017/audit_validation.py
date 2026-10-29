"""One validation pass; retain negative frames and reuse Ultralytics AP matching."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import DetMetrics

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY = ROOT / 'experiments/baselines_100e_20261017'
INDEX = None


def summarize(stats, selected):
    selected = list(selected)
    nt = sum(len(stats['target_cls'][i]) for i in selected)
    negatives = [i for i in selected if len(stats['target_cls'][i]) == 0]
    result = dict(frames=len(selected), objects=nt, negative_frames=len(negatives))
    for threshold in [0.1, 0.25]:
        counts = [int((stats['conf'][i] >= threshold).sum()) for i in negatives]
        positive_hits = sum(int(np.any((stats['conf'][i] >= threshold) & stats['tp'][i][:, 0]))
                            for i in selected if len(stats['target_cls'][i]))
        positive_frames = len(selected) - len(negatives)
        result[f'conf_{threshold}'] = dict(
            background_frames_with_fp=sum(c > 0 for c in counts),
            background_frame_fp_rate=sum(c > 0 for c in counts) / len(negatives) if negatives else None,
            fp_per_background_frame=sum(counts) / len(negatives) if negatives else None,
            positive_frame_hit_rate_iou50=positive_hits / positive_frames if positive_frames else None)
    if nt and selected:
        metric = DetMetrics(names={0: 'uav'})
        metric.stats = {k: [v[i] for i in selected] for k, v in stats.items()}
        metric.process()
        result.update({k: float(v) for k, v in metric.results_dict.items()})
    else:
        result.update({'metrics/mAP50(B)': None, 'metrics/mAP50-95(B)': None})
    result['small_positive_subset'] = 0 < nt < 20
    return result


class AuditValidator(DetectionValidator):
    def update_metrics(self, preds, batch):
        if not hasattr(self, 'frame_stems'):
            self.frame_stems = []
        super().update_metrics(preds, batch)
        self.frame_stems.extend(Path(p).stem for p in batch['im_file'])

    def get_stats(self):
        raw = self.metrics.stats
        assert len(self.frame_stems) == len(raw['conf']) == 9066
        assert set(self.frame_stems) == set(INDEX)
        groups = {'ALL': list(range(len(self.frame_stems))),
                  'POS_ONLY': [i for i, s in enumerate(self.frame_stems) if INDEX[s]['exist'] == 1],
                  'NEG_ONLY': [i for i, s in enumerate(self.frame_stems) if INDEX[s]['exist'] == 0]}
        for attr in ['TC', 'DBC', 'CDB', 'SV', 'OC', 'OV', 'FM']:
            groups[attr] = [i for i, s in enumerate(self.frame_stems) if INDEX[s]['attr'].get(attr) == 1]
        groups['TC_POS_ONLY'] = [i for i in groups['POS_ONLY'] if INDEX[self.frame_stems[i]]['attr'].get('TC') == 1]
        groups['NON_TC_POS_ONLY'] = [i for i in groups['POS_ONLY'] if INDEX[self.frame_stems[i]]['attr'].get('TC') == 0]
        for size in [1, 2, 3, 4]:
            groups[f'SIZE_CODE_{size}_POS_ONLY'] = [i for i in groups['POS_ONLY']
                                                  if INDEX[self.frame_stems[i]]['attr'].get('SIZE') == size]
        self.audit_groups = {name: summarize(raw, ids) for name, ids in groups.items()}
        flattened = {k: np.concatenate(v, axis=0) for k, v in raw.items()}
        flattened['pred_image_idx'] = np.concatenate([np.full(len(c), i, np.int32) for i, c in enumerate(raw['conf'])])
        flattened['target_image_idx'] = np.concatenate([np.full(len(c), i, np.int32) for i, c in enumerate(raw['target_cls'])])
        flattened['frame_stems'] = np.asarray(self.frame_stems)
        np.savez_compressed(self.save_dir / 'frame_stats.npz', **flattened)
        result = super().get_stats()
        for key in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)']:
            assert abs(self.audit_groups['ALL'][key] - result[key]) < 1e-9
        (self.save_dir / 'groups.json').write_text(json.dumps(self.audit_groups, indent=2))
        return result


def main():
    global INDEX
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--weights', type=Path, required=True)
    args = ap.parse_args()
    INDEX = json.loads((ROOT / 'data/yolo/attribute_index.json').read_text())['val']
    output = ROOT / 'runs/baselines_100e_20261017/validation_audit' / args.name
    assert not output.exists(), output
    model = YOLO(str(args.weights))
    metrics = model.val(validator=AuditValidator, data=str(STUDY / 'cst_original.yaml'),
                        split='val', imgsz=640, batch=128, workers=4, device='0', half=True,
                        conf=0.001, iou=0.7, max_det=300, rect=True, plots=False,
                        save_json=False, project=str(output.parent), name=args.name,
                        exist_ok=False, verbose=False)
    report = dict(name=args.name, weights=str(args.weights),
                  weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                  protocol=dict(split='val', frames=9066, includes_negatives=True,
                                batch=128, half=True, rect=True, imgsz=640,
                                conf=0.001, nms_iou=0.7, max_det=300,
                                ap_implementation='Ultralytics DetMetrics',
                                no_training=True),
                  metrics={k: float(v) for k, v in metrics.results_dict.items()},
                  groups=json.loads((output / 'groups.json').read_text()))
    (output / 'audit.json').write_text(json.dumps(report, indent=2))
    print('AUDIT_COMPLETE', args.name, report['metrics'], flush=True)


if __name__ == '__main__':
    main()
