"""Freeze original CST split lists and the previous YOLO11n training recipe."""
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY = ROOT / 'experiments/baselines_100e_20261017'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    STUDY.mkdir(parents=True, exist_ok=True)
    reset = json.loads((ROOT / 'baseline_reset_latest.json').read_text())
    source = Path(reset['old_runs']) / 'baseline_yolo11n/args.yaml'
    old = yaml.safe_load(source.read_text())
    (STUDY / 'reference_yolo11n_args.yaml').write_text(source.read_text())
    index_path = ROOT / 'data/yolo/attribute_index.json'
    index = json.loads(index_path.read_text())
    split_info = {}
    identities = {}
    for split in ['train', 'val', 'test']:
        paths = []
        label_hash = hashlib.sha256()
        objects = negatives = 0
        identities[split] = set()
        for stem, rec in sorted(index[split].items()):
            image = ROOT / 'data/yolo' / split / 'images' / (stem + '.jpg')
            label = ROOT / 'data/yolo' / split / 'labels' / (stem + '.txt')
            assert image.is_file() and label.is_file(), stem
            assert '__aat' not in stem, stem
            stat = image.stat()
            identities[split].add((stat.st_dev, stat.st_ino))
            text = label.read_text()
            rows = [line for line in text.splitlines() if line.strip()]
            objects += len(rows)
            negatives += not bool(rows)
            if rec['exist'] == 0:
                assert not rows, stem
            label_hash.update((stem + '\n' + text + '\n').encode())
            paths.append(str(image))
        manifest = STUDY / (split + '.txt')
        manifest.write_text('\n'.join(paths) + '\n')
        split_info[split] = dict(images=len(paths), objects=objects,
                                 negatives=negatives, list_sha256=digest(manifest),
                                 label_sha256=label_hash.hexdigest())
    assert not identities['train'] & identities['val']
    assert not identities['train'] & identities['test']
    assert not identities['val'] & identities['test']
    data = dict(path=str(ROOT / 'data/yolo'), train=str(STUDY / 'train.txt'),
                val=str(STUDY / 'val.txt'), test=str(STUDY / 'test.txt'),
                nc=1, names=['uav'])
    (STUDY / 'cst_original.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
    excluded_keys = {'task', 'mode', 'model', 'data', 'project', 'name',
                     'save_dir', 'resume', 'cfg'}
    common = {k: v for k, v in old.items()
              if k in DEFAULT_CFG_DICT and k not in excluded_keys}
    common.update(epochs=100, patience=20, imgsz=640, batch=256, workers=16,
                  seed=0, device='0', cache=False, pretrained=True,
                  deterministic=True, resume=False, exist_ok=False,
                  data=str(STUDY / 'cst_original.yaml'),
                  project=str(ROOT / 'runs/baselines_100e_20261017'))
    (STUDY / 'common_args.json').write_text(json.dumps(common, indent=2))
    os.chdir(ROOT)
    models = []
    for name in ['yolov8n', 'yolo11n', 'yolov5nu', 'yolo26n']:
        path = ROOT / (name + '.pt')
        model = YOLO(str(path) if path.exists() else name + '.pt')
        loaded = Path(model.ckpt_path).resolve()
        assert loaded.is_file()
        models.append(dict(name=name, weights=str(loaded), sha256=digest(loaded),
                           pretrained_parameters=sum(p.numel() for p in model.model.parameters())))
        print('WEIGHT_READY', models[-1], flush=True)
        del model
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       ultralytics=ultralytics.__version__, cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(0))
    contract = dict(models=models, splits=split_info, index_sha256=digest(index_path),
                    reference_args_sha256=digest(source),
                    common_args_sha256=digest(STUDY / 'common_args.json'),
                    environment=environment, max_parallel=2,
                    stopping='100 maximum epochs, patience 20 on framework validation fitness',
                    runtime=dict(workers=16, prefetch_factor=1, pin_memory=False,
                                 multiprocessing_context='forkserver'),
                    unknown_legacy_keys=[k for k in old if k not in DEFAULT_CFG_DICT],
                    change_from_reference='Only model/run paths and original split manifests; runtime prefetch/pinning/context bounded for two concurrent jobs. No AAT links, FP loss, distillation or custom augmentation.')
    (STUDY / 'contract.json').write_text(json.dumps(contract, indent=2))
    print('PREPARED', json.dumps(contract, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
