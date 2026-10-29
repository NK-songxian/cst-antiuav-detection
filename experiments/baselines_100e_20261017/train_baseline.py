"""One unmodified YOLO model with bounded data prefetch and auditable completion."""
import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--attempt', type=int, default=1)
    args = ap.parse_args()
    study = args.study.resolve()
    contract = json.loads((study / 'contract.json').read_text())
    job = next(m for m in contract['models'] if m['name'] == args.name)
    common = json.loads((study / 'common_args.json').read_text())
    run_name = args.name if args.attempt == 1 else f'{args.name}_attempt{args.attempt}'
    run = Path(common['project']) / run_name
    assert not run.exists(), f'Will not overwrite {run}'
    for split, rec in contract['splits'].items():
        assert hashlib.sha256((study / (split + '.txt')).read_bytes()).hexdigest() == rec['list_sha256']
    assert hashlib.sha256(Path(job['weights']).read_bytes()).hexdigest() == job['sha256']
    # Limit only input scheduling; training hyperparameters and transforms are unchanged.
    # Forkserver workers do not inherit the main process's initialized CUDA context.
    import torch
    from ultralytics import YOLO
    from ultralytics.data import build
    base_loader = build.InfiniteDataLoader

    class BoundedLoader(base_loader):
        def __init__(self, *pos, **kwargs):
            if kwargs.get('num_workers', 0) > 0:
                kwargs['prefetch_factor'] = 1
                kwargs['multiprocessing_context'] = 'forkserver'
            kwargs['pin_memory'] = False
            super().__init__(*pos, **kwargs)

    build.InfiniteDataLoader = BoundedLoader
    common.update(name=run_name)
    evidence = study / 'jobs' / run_name
    evidence.mkdir(parents=True, exist_ok=False)
    manifest = dict(name=args.name, attempt=args.attempt, started_at=now(), status='running',
                    pid=os.getpid(), argv=sys.argv, config=common,
                    weights=job, environment=contract['environment'],
                    runtime=contract['runtime'], actual_model_parameters=None)
    manifest_path = evidence / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    model = None
    try:
        model = YOLO(job['weights'])
        def model_ready(trainer):
            manifest['actual_model_parameters'] = sum(p.numel() for p in trainer.model.parameters())
            manifest['actual_optimizer'] = type(trainer.optimizer).__name__
            manifest['actual_train_images'] = len(trainer.train_loader.dataset)
            manifest['actual_val_images'] = len(trainer.test_loader.dataset)
            manifest_path.write_text(json.dumps(manifest, indent=2))
        model.add_callback('on_train_start', model_ready)
        model.train(**common)
        assert (run / 'weights/best.pt').is_file(), 'Missing best checkpoint'
        assert (run / 'results.csv').is_file(), 'Missing epoch results'
        manifest.update(status='completed', ended_at=now(), last_epoch=int(model.trainer.epoch) + 1,
                        output=str(run), completion_reason='training_returned_successfully')
    except BaseException as exc:
        manifest.update(status='failed', ended_at=now(), error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if model is not None and getattr(model, 'trainer', None) is not None:
            for loader in [getattr(model.trainer, 'train_loader', None), getattr(model.trainer, 'test_loader', None)]:
                iterator = getattr(loader, 'iterator', None)
                if iterator is not None and hasattr(iterator, '_shutdown_workers'):
                    try:
                        iterator._shutdown_workers()
                    except Exception:
                        pass
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
