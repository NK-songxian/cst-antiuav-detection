"""Prepare paired AAT sampling previews with matched TC dose and exact epoch length.

Preview only: a training implementation must rotate the draw each epoch and audit
the actual DataLoader sampler. Existing dataset folders and labels stay unchanged.
"""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
OUT = ROOT / 'experiments/innovation_plan_20261017'


def draw(tc, other, n_tc, rng):
    assert len(tc) <= n_tc <= 2 * len(tc)
    extra = n_tc - len(tc)
    # At most two appearances per TC frame, avoiding a handful of repeated frames.
    chosen = list(tc) + rng.sample(tc, extra)
    chosen += rng.sample(other, len(tc) + len(other) - n_tc)
    return chosen


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index_path = ROOT / 'data/yolo/attribute_index.json'
    idx = json.loads(index_path.read_text())['train']
    seqs = defaultdict(list)
    negatives = []
    for stem, rec in sorted(idx.items()):
        if rec['exist'] == 1:
            seqs[rec['seq']].append(stem)
        else:
            negatives.append(stem)
    rng = random.Random(20261017)
    quota = []
    per_sequence = {}
    for seq, stems in sorted(seqs.items()):
        tc = [s for s in stems if idx[s]['attr'].get('TC') == 1]
        other = [s for s in stems if idx[s]['attr'].get('TC') == 0]
        desired = round(len(stems) * 1.5 * len(tc) / (1.5 * len(tc) + len(other)))
        desired = max(len(tc), min(desired, 2 * len(tc), len(stems)))
        picked = draw(tc, other, desired, rng)
        quota.extend(picked)
        per_sequence[seq] = dict(positive_slots=len(stems), tc_original=len(tc), tc_planned=desired)
    tc_all = [s for stems in seqs.values() for s in stems if idx[s]['attr'].get('TC') == 1]
    other_all = [s for stems in seqs.values() for s in stems if idx[s]['attr'].get('TC') == 0]
    n_tc = sum(idx[s]['attr'].get('TC') == 1 for s in quota)
    global_picked = draw(tc_all, other_all, n_tc, random.Random(20261017))
    original = sorted(idx)
    schedules = {'baseline': original, 'A1_global_TC_dose_matched': global_picked + negatives,
                 'A2_sequence_quota_TC': quota + negatives}
    original_seq = Counter(idx[s]['seq'] for s in original)
    reports = {}
    for name, stems in schedules.items():
        rng.shuffle(stems)
        counts = Counter(stems)
        seq_counts = Counter(idx[s]['seq'] for s in stems)
        bg = [s for s in stems if idx[s]['exist'] == 0]
        assert len(stems) == 25612 and len(bg) == 3612 and set(bg) == set(negatives)
        assert max(counts.values()) <= 2
        if name == 'A2_sequence_quota_TC':
            assert seq_counts == original_seq
        text = '\n'.join(str(ROOT / 'data/yolo/train/images' / (s + '.jpg')) for s in stems) + '\n'
        (OUT / (name + '_epoch0_preview.txt')).write_text(text)
        reports[name] = dict(frames=len(stems), unique_frames=len(counts), negative_slots=len(bg),
                             tc_positive_slots=sum(idx[s]['exist'] == 1 and idx[s]['attr'].get('TC') == 1 for s in stems),
                             max_frame_repeats=max(counts.values()),
                             sequence_total_variation=0.5 * sum(abs(seq_counts[k] - v) for k, v in original_seq.items()) / len(stems),
                             exact_sequence_quotas=seq_counts == original_seq,
                             sha256=hashlib.sha256(text.encode()).hexdigest(),
                             seq_counts=dict(seq_counts))
    assert reports['A1_global_TC_dose_matched']['tc_positive_slots'] == reports['A2_sequence_quota_TC']['tc_positive_slots']
    report = dict(status='PLANNING_PREVIEW_NOT_TRAINING', index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
                  seed=20261017, local_tc_weight=1.5, models=['yolo11n'], reports=reports,
                  per_sequence=per_sequence,
                  runtime_requirement='Redraw each epoch using seed and epoch; keep all negative frames once, rotate omitted non-TC positives; audit actual loader indices and Mosaic secondary draws.')
    (OUT / 'sampling_preview.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({name: {k: v for k, v in rec.items() if k != 'seq_counts'} for name, rec in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
