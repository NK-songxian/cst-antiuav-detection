"""TCE revision training: T2 (ghost labels, nc=1) and T3 (ghost labels +
temporal channel classes, nc=3), frozen matched protocol from
cst_gap_confirmation_20261019 (same PNGs, sampler, SharedBrightness, batch=256,
native best.pt by val AP50-95).

For T3 the validator filters predictions to the current-frame class (class 2)
before matching, so training-time selection and every reported AP refer to the
deployable 'current' output only -- the comparable criterion against the single
class T1 reference.
"""
import argparse
import hashlib
import json
import os
import sys
import traceback
from copy import copy
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY = ROOT / 'experiments/tce_revision_20261109'
GAP = ROOT / 'experiments/cst_gap_confirmation_20261019'
sys.path.insert(0, str(GAP))
import matched_runtime as rt  # noqa: E402  (sets INPUT_MODE='temporal', imports ti)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class CurrentClassValidator(rt.MatchedValidator):
    """Filter predictions to the current-frame channel class before matching.

    Ultralytics 8.4.177 update_metrics receives dict preds:
    {'bboxes': (n,4), 'conf': (n,), 'cls': (n,), 'extra': (n,0)}.
    """
    CURRENT_CLS = 2

    @staticmethod
    def _filter(p, cls_id):
        keep = p['cls'] == cls_id
        return {'bboxes': p['bboxes'][keep], 'conf': p['conf'][keep],
                'cls': p['cls'][keep], 'extra': p['extra'][keep]}

    def update_metrics(self, preds, batch):
        preds = [self._filter(p, self.CURRENT_CLS) for p in preds]
        super().update_metrics(preds, batch)


def make_trainer_cls(validator_cls):
    class TCETrainer(rt.MatchedTrainer):
        def get_validator(self):
            args = copy(self.args)
            args.workers = 2
            return validator_cls(self.test_loader, save_dir=self.save_dir, args=args,
                                 _callbacks=self.callbacks)
    return TCETrainer


CURRENT_CLS = 2  # temporal class id of the deployable 'current-frame' channel


def apply_ghost_regression_exclusion():
    """TCE v2: decoupled temporal supervision.

    Ghost instances (classes 0/1) supervise ONLY the classification head: their
    anchors' box/DFL loss weight is zeroed and the box/DFL normalization uses
    the current-class score sum only, so the shared regression head receives
    the same per-anchor gradient scale as the nc=1 reference. Network
    architecture untouched.
    """
    import ultralytics.utils.loss as uloss

    orig_forward = uloss.BboxLoss.forward

    def forward_cls_only(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                         target_scores, target_scores_sum, fg_mask, imgsz, stride):
        cls_id = target_scores.argmax(-1)                     # (bs, anchors)
        keep = cls_id == CURRENT_CLS
        cur_scores = target_scores.clone()
        cur_scores[~keep] = 0.0
        cur_sum = cur_scores.sum().clamp_(min=1)
        return orig_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                            cur_scores, cur_sum, fg_mask, imgsz, stride)

    uloss.BboxLoss.forward = forward_cls_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['t2', 't3', 't3v2'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    name = f'TCE_{a.arm.upper()}_seed{a.seed}'
    run_root = ROOT / 'runs/tce_revision_20261109'
    job_dir = STUDY / 'jobs' / name
    job_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = job_dir / 'manifest.json'

    common = json.loads((GAP / 'common_args.json').read_text())
    common.update(
        name=name, seed=a.seed, workers=8,
        data=str(STUDY / f'cst_tce_{"t3" if a.arm == "t3v2" else a.arm}.yaml'),
        project=str(run_root))
    weights = ROOT / 'weights/yolo11n.pt'

    manifest = dict(name=name, arm=a.arm, seed=a.seed, status='running', started_at=now(),
        pid=os.getpid(), config=common, weights_sha256=sha(weights),
        data_ready_sha256=sha(STUDY / 'data_ready.json'),
        protocol=dict(epochs=common['epochs'], batch=256, imgsz=640, patience=20,
            selection='native best.pt by val AP50-95'
                      + (', validator filtered to current class 2' if a.arm != 't2' else ''),
            supervision=('ghost regression-exclusion: ghosts cls-only, box/DFL normalized '
                         'by current-class score sum' if a.arm == 't3v2'
                         else ('all ghosts as class uav' if a.arm == 't2'
                               else 'ghosts as separate temporal classes, full regression')),
            test_used_by_training=False))

    def save():
        tmp = manifest_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(manifest_path)

    save()
    torch.set_num_threads(2)
    cv2.setNumThreads(0)
    torch.cuda.set_per_process_memory_fraction(
        35970351104 / torch.cuda.get_device_properties(0).total_memory)
    rt.ti.ACTIVE_JOB = dict(run_name=name, sampling='uniform')
    import ultralytics.data.augment as aug
    aug.RandomHSV = rt.ti.SharedBrightness
    from ultralytics.data import build
    original = build.InfiniteDataLoader

    class BoundedLoader(original):
        def __init__(self, *pos, **kw):
            if kw.get('num_workers', 0) > 0:
                kw.update(prefetch_factor=1, multiprocessing_context='forkserver')
            kw['pin_memory'] = False
            super().__init__(*pos, **kw)
    build.InfiniteDataLoader = BoundedLoader

    validator_cls = CurrentClassValidator if a.arm != 't2' else rt.MatchedValidator
    trainer_cls = make_trainer_cls(validator_cls)
    if a.arm == 't3v2':
        apply_ghost_regression_exclusion()

    model = None
    try:
        model = YOLO(str(weights))

        def ready(t):
            assert t.batch_size == 256 and t.args.imgsz == 640
            assert len(t.train_loader.dataset) == 25612
            assert len(t.test_loader.dataset) == 9066
            manifest.update(actual_batch=t.batch_size,
                actual_train_frames=len(t.train_loader.dataset),
                actual_val_frames=len(t.test_loader.dataset),
                parameters=sum(p.numel() for p in t.model.parameters()),
                nc=int(t.model.model[-1].nc))
            save()

        def epoch_start(t):
            t.train_loader.sampler.set_epoch(t.epoch)

        def batch_end(t):
            assert t.batch_size == 256, 'Silent batch reduction invalidates the comparison'

        def epoch_end(t):
            manifest.update(latest_epoch=int(t.epoch) + 1,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), updated_at=now())
            save()

        model.add_callback('on_train_start', ready)
        model.add_callback('on_train_epoch_start', epoch_start)
        model.add_callback('on_train_batch_end', batch_end)
        model.add_callback('on_train_epoch_end', epoch_end)
        model.train(trainer=trainer_cls, **common)
        run = run_root / name
        assert (run / 'weights/best.pt').is_file()
        manifest.update(status='trained', ended_training_at=now(),
            best_sha256=sha(run / 'weights/best.pt'))
        save()
    except BaseException as e:
        manifest.update(status='failed', ended_at=now(), error=repr(e),
            traceback=traceback.format_exc())
        save()
        raise
    finally:
        if model is not None and getattr(model, 'trainer', None) is not None:
            for loader in [getattr(model.trainer, 'train_loader', None),
                           getattr(model.trainer, 'test_loader', None)]:
                if isinstance(loader, rt.ti.EpochLoader):
                    loader.close()
                else:
                    it = getattr(loader, 'iterator', None)
                    if it is not None and hasattr(it, '_shutdown_workers'):
                        try:
                            it._shutdown_workers()
                        except Exception:
                            pass
        save()


if __name__ == '__main__':
    main()
