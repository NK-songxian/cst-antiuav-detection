"""Prepare causal held-out inputs and evaluate every frozen model sequentially."""
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY = ROOT/'experiments/aat_temporal_100e_20261017'
OUT = STUDY/'completed_review_20261017'
sys.path.insert(0, str(STUDY))
from sampling import build_history


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    cv2.setNumThreads(1)
    index = json.loads((ROOT/'data/yolo/attribute_index.json').read_text())
    seqs = {s: {r['seq'] for r in rec.values()} for s, rec in index.items()}
    assert not (seqs['train'] & seqs['test'] or seqs['val'] & seqs['test'] or seqs['train'] & seqs['val'])
    history = build_history(index['test'])
    assert len(history) == 14446
    for stem, triplet in history.items():
        assert triplet[-1] == stem
        assert all(index['test'][s]['seq'] == index['test'][stem]['seq'] and
                   index['test'][s]['frame'] <= index['test'][stem]['frame'] for s in triplet)
    data = OUT/'temporal_test'
    (data/'images').mkdir(parents=True, exist_ok=False)
    (data/'labels').mkdir()
    source = ROOT/'data/yolo/test'

    def make(stem):
        triplet = history[stem]
        ims = {s: cv2.imread(str(source/'images'/(s+'.jpg')), cv2.IMREAD_GRAYSCALE) for s in set(triplet)}
        assert all(im is not None for im in ims.values())
        assert len({im.shape for im in ims.values()}) == 1
        saved = data/'images'/(stem+'.png')
        assert cv2.imwrite(str(saved), cv2.merge([ims[s] for s in triplet]), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        (data/'labels'/(stem+'.txt')).symlink_to(source/'labels'/(stem+'.txt'))
        return str(saved), sha(saved)

    stems = sorted(index['test'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        outputs = list(pool.map(make, stems))
    testlist = OUT/'temporal_test.txt'
    testlist.write_text('\n'.join(p for p, h in outputs)+'\n')
    (OUT/'history_test.json').write_text(json.dumps(history))
    (OUT/'temporal_test_image_hashes.txt').write_text(''.join(s+' '+h+'\n' for s, (_, h) in zip(stems, outputs)))
    for stem in stems[::73]:
        saved = cv2.imread(str(data/'images'/(stem+'.png')))
        for channel, src in enumerate(history[stem]):
            assert np.array_equal(saved[:, :, channel], cv2.imread(str(source/'images'/(src+'.jpg')), cv2.IMREAD_GRAYSCALE))
        assert (data/'labels'/(stem+'.txt')).read_bytes() == (source/'labels'/(stem+'.txt')).read_bytes()
    cfg = yaml.safe_load((STUDY/'cst_temporal.yaml').read_text())
    cfg['test'] = str(testlist)
    (OUT/'cst_temporal_evaluation.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    ready = dict(frames=len(stems), sequences=len(seqs['test']),
        negatives=sum(not r['exist'] for r in index['test'].values()),
        full_history=sum(len(set(h)) == 3 for h in history.values()),
        history_uses_labels=False, future_frames_used=False, splits_sequence_disjoint=True,
        raw_current_channel_is_lossless=True, history_sha256=sha(OUT/'history_test.json'))
    (OUT/'test_data_audit.json').write_text(json.dumps(ready, indent=2))
    print('DATA_READY', json.dumps(ready), flush=True)
    # Freeze model list before any held-out scores become available.
    tasks = [('C1_temporal_uniform', 'val', False)]
    tasks += [(n, 'test', False) for n in ['yolo11n', 'A1_aat_global', 'A2_aat_sequence',
        'C1_temporal_uniform', 'C2_temporal_aat_sequence', 'yolov8n', 'yolov5nu', 'yolo26n']]
    tasks += [(n, 'test', True) for n in ['C1_temporal_uniform', 'C2_temporal_aat_sequence']]
    manifest = dict(started_at=datetime.now(timezone.utc).isoformat(), tasks=tasks, completed=[],
        checkpoint_policy='All native best.pt; no test-based selection', exploratory_single_seed=True,
        controls='Repeat-current is a post-training diagnostic; distribution shift limits interpretation',
        env={k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS']})
    (OUT/'evaluation_plan.json').write_text(json.dumps(manifest, indent=2))
    for name, split, repeat in tasks:
        mem = int(Path('/sys/fs/cgroup/memory.current').read_text())
        limit = int(Path('/sys/fs/cgroup/memory.max').read_text())
        assert limit-mem > 16*1024**3, 'Insufficient container RAM headroom'
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
        assert int(gpu.strip().splitlines()[0]) > 14*1024, 'Insufficient GPU headroom'
        tag = f'{split}_{name}'+('_repeat_current' if repeat else '')
        cmd = [sys.executable, str(OUT/'evaluate_cst_completed.py'), '--name', name, '--split', split]
        if repeat:
            cmd += ['--repeat-current']
        print('EVALUATING', tag, flush=True)
        with (OUT/(tag+'.log')).open('w') as log:
            ret = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        manifest['completed'].append(dict(task=tag, returncode=ret.returncode, ended_at=datetime.now(timezone.utc).isoformat()))
        (OUT/'evaluation_status.json').write_text(json.dumps(manifest, indent=2))
        assert ret.returncode == 0, (tag, ret.returncode)
    manifest.update(status='completed', ended_at=datetime.now(timezone.utc).isoformat())
    (OUT/'evaluation_status.json').write_text(json.dumps(manifest, indent=2))
    print('ALL_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
