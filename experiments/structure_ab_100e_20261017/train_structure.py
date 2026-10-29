"""A/B training on the frozen CST manifests, with checkpoint deployment export."""
import argparse
import copy
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.data import build
from ultralytics.nn.tasks import DetectionModel

from structure_feasibility import FoldableContrastDW
from structure_trainer import TaskATrainer, TaskBTrainer


def now():
    return datetime.now(timezone.utc).isoformat()


def export_deployment(best, destination):
    ckpt = torch.load(best, map_location='cpu', weights_only=False)
    model = (ckpt.get('ema') or ckpt['model']).float().eval()
    fused = copy.deepcopy(model).fuse(verbose=False).eval()
    x = torch.rand(1, 3, 160, 160)
    with torch.no_grad():
        a, b = model(x)[0], fused(x)[0]
    torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-4)
    error = float((a-b).abs().max())
    # Remove all custom classes from the deployment artifact.
    for parent in list(fused.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, FoldableContrastDW):
                parent._modules[name] = nn.Sequential(child.reparam, child.act)
    fused.__class__ = DetectionModel
    ckpt.update(model=fused, ema=None, optimizer=None, epoch=-1)
    torch.save(ckpt, destination)
    # Public loader must accept the deployment checkpoint, not just torch.load.
    restored = YOLO(str(destination)).model.float().eval()
    with torch.no_grad():
        c = restored(x)[0]
    torch.testing.assert_close(b, c, atol=1e-3, rtol=1e-4)
    return dict(path=str(destination), parameters=sum(p.numel() for p in restored.parameters()),
                whole_model_folding_max_abs_error=error, reload_pass=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--name', choices=['A_zero_deploy', 'B_p2_compact'], required=True)
    ap.add_argument('--attempt', type=int, default=1)
    args = ap.parse_args()
    study = args.study.resolve()
    contract = json.loads((study / 'contract.json').read_text())
    common = json.loads((study / 'common_args.json').read_text())
    job = next(j for j in contract['models'] if j['name'] == args.name)
    run_name = args.name if args.attempt == 1 else f'{args.name}_attempt{args.attempt}'
    run = Path(common['project']) / run_name
    assert not run.exists(), run
    for split, record in contract['splits'].items():
        p = Path(contract['baseline_study']) / (split + '.txt')
        assert hashlib.sha256(p.read_bytes()).hexdigest() == record['list_sha256']
    assert hashlib.sha256(Path(job['weights']).read_bytes()).hexdigest() == job['sha256']
    assert (study / 'preflight.json').is_file()
    assert all((study / ('memory_' + n + '.json')).is_file() for n in ['A_zero_deploy', 'B_p2_compact'])
    base_loader = build.InfiniteDataLoader

    class BoundedLoader(base_loader):
        def __init__(self, *pos, **kwargs):
            if kwargs.get('num_workers', 0) > 0:
                kwargs['prefetch_factor'] = 1
                kwargs['multiprocessing_context'] = 'forkserver'
            kwargs['pin_memory'] = False
            super().__init__(*pos, **kwargs)

    build.InfiniteDataLoader = BoundedLoader
    evidence = study / 'jobs' / run_name
    evidence.mkdir(parents=True, exist_ok=False)
    common.update(name=run_name)
    manifest = dict(status='running', name=args.name, started_at=now(), pid=os.getpid(),
                    attempt=args.attempt, config=common, weights=job,
                    runtime=contract['runtime'], environment=contract['environment'],
                    baseline_study=contract['baseline_study'])
    mp = evidence / 'manifest.json'
    mp.write_text(json.dumps(manifest, indent=2))
    model = None
    try:
        model = YOLO(job['weights'])

        def ready(trainer):
            m = trainer.model
            manifest.update(actual_model_parameters=sum(p.numel() for p in m.parameters()),
                            actual_train_images=len(trainer.train_loader.dataset),
                            actual_val_images=len(trainer.test_loader.dataset),
                            actual_optimizer=type(trainer.optimizer).__name__,
                            strides=m.stride.tolist(), transfer=m.structure_transfer,
                            replacements=m.structure_replacements)
            (evidence / 'constructed_model.yaml.json').write_text(json.dumps(m.yaml, indent=2))
            mp.write_text(json.dumps(manifest, indent=2))

        def epoch_done(trainer):
            values = [float(m.contrast_bn.weight.detach().abs().mean())
                      for m in trainer.model.modules() if isinstance(m, FoldableContrastDW)]
            manifest.update(last_epoch_seen=int(trainer.epoch)+1, contrast_bn_abs_mean=values)
            mp.write_text(json.dumps(manifest, indent=2))

        model.add_callback('on_train_start', ready)
        model.add_callback('on_fit_epoch_end', epoch_done)
        model.train(trainer=TaskATrainer if args.name == 'A_zero_deploy' else TaskBTrainer, **common)
        assert (run / 'weights/best.pt').is_file() and (run / 'results.csv').is_file()
        manifest['deployment'] = export_deployment(run / 'weights/best.pt', run / 'weights/deploy.pt')
        manifest.update(status='completed', ended_at=now(), output=str(run),
                        last_epoch=int(model.trainer.epoch)+1,
                        completion_reason='training_and_deployment_reload_passed')
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
        mp.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
