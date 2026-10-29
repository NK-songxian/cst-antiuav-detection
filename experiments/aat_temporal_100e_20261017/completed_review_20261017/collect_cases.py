"""Select both improvements and failures, retaining frame-level selection reasons."""
import json
import shutil
from pathlib import Path
import numpy as np

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
OUT = ROOT/'experiments/aat_temporal_100e_20261017/completed_review_20261017'
NAMES = ['yolo11n', 'C1_temporal_uniform', 'C2_temporal_aat_sequence']


def main():
    index = json.loads((ROOT/'data/yolo/attribute_index.json').read_text())['test']
    history = json.loads((OUT/'history_test.json').read_text())
    frames = {}
    predictions = {}
    for name in NAMES:
        path = OUT/'evaluations/test'/name
        d = np.load(path/'frame_stats.npz')
        scores, hits = np.full(len(d['frame_stems']), -np.inf), np.full(len(d['frame_stems']), -np.inf)
        np.maximum.at(scores, d['pred_image_idx'], d['conf'])
        np.maximum.at(hits, d['pred_image_idx'], np.where(d['tp'][:, 0], d['conf'], -np.inf))
        frames[name] = {str(s): {'score': float(v), 'hit': float(h)} for s, v, h in zip(d['frame_stems'], scores, hits)}
        predictions[name] = json.loads((path/'predictions_conf010.json').read_text())
    base, c1, c2 = [frames[n] for n in NAMES]
    available = sorted(index)
    predicates = [
        ('temporal_recovers_target', lambda s: index[s]['exist'] and base[s]['hit'] < .25 and c1[s]['hit'] >= .25 and c2[s]['hit'] >= .25),
        ('temporal_removes_background_fp', lambda s: not index[s]['exist'] and base[s]['score'] >= .25 and c1[s]['score'] < .25 and c2[s]['score'] < .25),
        ('remaining_temporal_background_fp', lambda s: not index[s]['exist'] and (c1[s]['score'] >= .25 or c2[s]['score'] >= .25)),
        ('temporal_regression', lambda s: index[s]['exist'] and base[s]['hit'] >= .25 and (c1[s]['hit'] < .25 or c2[s]['hit'] < .25))]
    dest = OUT/'qualitative'
    dest.mkdir(exist_ok=True)
    image_dir = dest/'images'
    image_dir.mkdir(exist_ok=True)
    chosen = []
    used_sequences = set()
    for kind, predicate in predicates:
        candidates = [s for s in available if predicate(s)]
        # Lexicographically first available sequence/frame, not a hand-chosen confidence peak.
        selected = next((s for s in candidates if index[s]['seq'] not in used_sequences), candidates[0] if candidates else None)
        if selected is None:
            chosen.append(dict(kind=kind, matching_frames=0, selected=None))
            continue
        used_sequences.add(index[selected]['seq'])
        for stem in history[selected]:
            shutil.copy2(ROOT/'data/yolo/test/images'/(stem+'.jpg'), image_dir/(stem+'.jpg'))
        label = (ROOT/'data/yolo/test/labels'/(selected+'.txt')).read_text()
        chosen.append(dict(kind=kind, matching_frames=len(candidates), selected=selected,
            metadata=index[selected], history=history[selected], yolo_ground_truth=label,
            models={n: dict(predictions=predictions[n][selected], frame_stats=frames[n][selected]) for n in NAMES}))
    (dest/'cases.json').write_text(json.dumps(dict(
        selection='Fixed confidence0.25; first lexicographic qualifying frame with sequence diversity. Selected after outcomes for illustration; not prevalence estimates.',
        colors='Ground truth cyan; predictions orange', cases=chosen), indent=2))
    print('CASES_COMPLETE', json.dumps([{k:v for k,v in r.items() if k in ['kind','matching_frames','selected']} for r in chosen]))


if __name__ == '__main__':
    main()
