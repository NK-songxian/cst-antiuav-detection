"""Materialize TCS delta-ablation datasets: delta1=[t-2,t-1,t], delta10=[t-20,t-10,t].

Mirrors aat_temporal_100e_20261017/prepare_study.py:temporal(): grayscale frames
merged channel-wise (BGR file order = past-to-current), clamp missing history
toward the current frame, labels symlinked from data/yolo, probe verification.
"""
import concurrent.futures
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY = ROOT / 'experiments/tcs_ablation_20261017'
VARIANTS = {'far15': (5, 15), 'far20': (5, 20)}


def sha(path):
    import hashlib
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def build_history(index, near, far):
    groups = defaultdict(dict)
    for stem, rec in index.items():
        groups[rec['seq']][rec['frame']] = stem
    result = {}
    for stem, rec in index.items():
        lookup = groups[rec['seq']]
        frame = rec['frame']
        p_near = lookup.get(frame - near, stem)
        p_far = lookup.get(frame - far, p_near)
        result[stem] = [p_far, p_near, stem]
    return result


def main():
    cv2.setNumThreads(1)
    index = json.loads((ROOT / 'data/yolo/attribute_index.json').read_text())
    for vname, (near, far) in VARIANTS.items():
        out = STUDY / 'data' / vname
        for split in ('train', 'val'):
            idx = index[split]
            history = build_history(idx, near, far)
            current = out / split
            (current / 'images').mkdir(parents=True, exist_ok=True)
            (current / 'labels').mkdir(exist_ok=True)
            source = ROOT / 'data/yolo' / split

            def make(stem):
                triplet = history[stem]
                ims = {s: cv2.imread(str(source / 'images' / (s + '.jpg')), cv2.IMREAD_GRAYSCALE)
                       for s in set(triplet)}
                assert all(im is not None for im in ims.values()), stem
                merged = cv2.merge([ims[s] for s in triplet])
                dest = current / 'images' / (stem + '.png')
                assert cv2.imwrite(str(dest), merged, [cv2.IMWRITE_PNG_COMPRESSION, 1]), dest
                label = current / 'labels' / (stem + '.txt')
                if not label.exists():
                    label.symlink_to(source / 'labels' / (stem + '.txt'))
                return stem, sha(dest)

            stems = sorted(idx)
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
                outputs = list(pool.map(make, stems))
            (out / f'temporal_{split}.txt').write_text(
                '\n'.join(str(current / 'images' / (s + '.png')) for s, _ in outputs) + '\n')
            (out / f'history_{split}.json').write_text(json.dumps(history))
            probes = stems[::max(1, len(stems) // 120)]
            for stem in probes:
                saved = cv2.imread(str(current / 'images' / (stem + '.png')))
                for ch, prev in enumerate(history[stem]):
                    raw = cv2.imread(str(source / 'images' / (prev + '.jpg')), cv2.IMREAD_GRAYSCALE)
                    assert np.array_equal(saved[:, :, ch], raw), stem
            print(f'[{vname}:{split}] {len(stems)} frames, clamped='
                  f'{sum(len(set(v)) < 3 for v in history.values())}', flush=True)
        yaml_text = (f"path: {out}\n"
                     f"train: {out / 'train' / 'temporal_train.txt'}\n"
                     f"val: {out / 'val' / 'temporal_val.txt'}\n"
                     f"nc: 1\nnames:\n- uav\n")
        (STUDY / f'cst_{vname}.yaml').write_text(yaml_text)
    (STUDY / 'data_ready.json').write_text(json.dumps(
        dict(status='PASS', variants=list(VARIANTS),
             created_at=datetime.now(timezone.utc).isoformat()), indent=2))
    print('TCS_ABLATION_DATA_DONE')


if __name__ == '__main__':
    main()
