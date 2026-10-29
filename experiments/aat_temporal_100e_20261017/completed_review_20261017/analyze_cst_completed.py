"""Compare fixed checkpoints, validation-calibrated thresholds and sequence bootstrap."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from ultralytics.utils.metrics import compute_ap

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY = ROOT/'experiments/aat_temporal_100e_20261017'
OUT = STUDY/'completed_review_20261017'
NAMES = ['yolov8n', 'yolo11n', 'yolov5nu', 'yolo26n', 'A1_aat_global',
         'A2_aat_sequence', 'C1_temporal_uniform', 'C2_temporal_aat_sequence']
MAIN = NAMES[4:]
K50, K95 = 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def valdir(name):
    if name == 'C1_temporal_uniform':
        return OUT/'evaluations/val'/name
    study = 'baselines_100e_20261017' if name in NAMES[:4] else 'aat_temporal_100e_20261017'
    return ROOT/'runs'/study/'validation_audit'/name


def per_frame(path):
    d = np.load(path/'frame_stats.npz')
    n = len(d['frame_stems'])
    pos = np.zeros(n, dtype=bool)
    pos[d['target_image_idx']] = True
    scores, hits = np.full(n, -np.inf), np.full(n, -np.inf)
    np.maximum.at(scores, d['pred_image_idx'], d['conf'])
    np.maximum.at(hits, d['pred_image_idx'], np.where(d['tp'][:, 0], d['conf'], -np.inf))
    return d, pos, scores, hits


def prepare_boot(path, index, sequences):
    d = np.load(path/'frame_stats.npz')
    codes = {s: i for i, s in enumerate(sequences)}
    seq = np.asarray([codes[index[str(s)]['seq']] for s in d['frame_stems']])
    order = np.argsort(-d['conf'])
    return dict(tp=d['tp'][order, 0], predseq=seq[d['pred_image_idx']][order],
                targets=np.bincount(seq[d['target_image_idx']], minlength=len(sequences)))


def weighted_ap50(prep, weights):
    w = weights[prep['predseq']]
    keep = w > 0
    w, tp = w[keep], prep['tp'][keep]
    nt = float(weights @ prep['targets'])
    if not nt:
        return np.nan
    if not len(w):
        return 0.0
    # Integer sequence weights exactly repeat each prediction; preserve the
    # fixed confidence sort to avoid adding random tie-breaking per bootstrap.
    tp = np.repeat(tp, w)
    tpc = tp.cumsum()
    return float(compute_ap(tpc/(nt+1e-16), tpc/np.arange(1, len(tp)+1))[0])


def main():
    index = json.loads((ROOT/'data/yolo/attribute_index.json').read_text())
    report = dict(as_of=datetime.now(timezone.utc).isoformat(),
        protocol='Native val-AP50-95 best.pt; fixed conf=0.001, IoU=0.7, max_det=300, batch128, rect640, FP16; all background frames included.',
        split_counts={s: dict(frames=len(v), objects=sum(r['exist'] for r in v.values()),
                             negatives=sum(not r['exist'] for r in v.values()), sequences=len({r['seq'] for r in v.values()})) for s, v in index.items()},
        uncertainty='One exploratory training seed; sequence bootstrap quantifies held-out sequence sampling uncertainty conditional on these weights, not training-seed variability.',
        models={}, diagnostics={}, selection_note='No checkpoint or detection threshold selected on test.',
        analysis_decisions=[dict(phase='POST_TRAINING_PRE_TEST', decision='Use all native best.pt, repeat-current diagnostics and val-calibrated FP thresholds.'),
                            dict(phase='POST_RESULTS', decision='Sequence bootstrap of five AP50 contrasts; descriptive sensitivity analysis, not preregistered confirmation.')])
    for name in NAMES:
        vd, td = valdir(name), OUT/'evaluations/test'/name
        va = json.loads((vd/'audit.json').read_text())
        ta = json.loads((td/'audit.json').read_text())
        v, pos, scores, hits = per_frame(vd)
        budget = int(np.floor((~pos).sum()*0.05))
        ordered = np.sort(scores[~pos])[::-1]
        threshold = float(np.nextafter(ordered[budget], np.inf))
        _, tpos, tscores, thits = per_frame(td)
        run = ROOT/'runs'/('baselines_100e_20261017' if name in NAMES[:4] else 'aat_temporal_100e_20261017')/name
        rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader((run/'results.csv').open())]
        best95, best50 = max(rows, key=lambda r: r[K95]), max(rows, key=lambda r: r[K50])
        row = dict(val=va['metrics'], test=ta['metrics'],
            val_background_fp025=va['groups']['ALL']['conf_0.25'],
            test_background_fp025=ta['groups']['ALL']['conf_0.25'],
            test_groups=ta['groups'], val_groups=va['groups'],
            epochs=len(rows), best95_epoch=int(best95['epoch']), peak_training_val_ap50=best50[K50],
            peak_training_val_ap50_epoch=int(best50['epoch']),
            parameters_before_fusion=ta['parameters_before_fusion'],
            validation_calibrated_5pct_fpr=dict(threshold=threshold,
                val_background_fpr=float(np.mean(scores[~pos] >= threshold)),
                val_positive_hit=float(np.mean(hits[pos] >= threshold)),
                test_background_fpr=float(np.mean(tscores[~tpos] >= threshold)),
                test_positive_hit=float(np.mean(thits[tpos] >= threshold)),
                note='Threshold fitted only on val; test FPR is unconstrained, not retuned.'),
            provenance=dict(weights_sha256=ta['weights_sha256'], test_audit=str(td/'audit.json'),
                            test_audit_sha256=sha(td/'audit.json'), stats_sha256=sha(td/'frame_stats.npz'),
                            val_audit=str(vd/'audit.json'), val_audit_sha256=sha(vd/'audit.json')))
        report['models'][name] = row
    baseline = report['models']['yolo11n']
    for name, row in report['models'].items():
        row['test_delta_ap50_pp'] = 100*(row['test'][K50]-baseline['test'][K50])
        row['val_delta_ap50_pp'] = 100*(row['val'][K50]-baseline['val'][K50])
        row['guardrails'] = {
            'val_background_fpr_increase_le_2pp': row['val_background_fp025']['background_frame_fp_rate'] <= baseline['val_background_fp025']['background_frame_fp_rate']+0.02,
            'val_standard_recall_drop_le_1pp': row['val']['metrics/recall(B)'] >= baseline['val']['metrics/recall(B)']-0.01}
    for name in ['C1_temporal_uniform', 'C2_temporal_aat_sequence']:
        a = json.loads((OUT/'evaluations/test'/(name+'_repeat_current')/'audit.json').read_text())
        report['diagnostics'][name] = dict(repeat_current_test_metrics=a['metrics'],
            repeat_current_background_fp025=a['groups']['ALL']['conf_0.25'],
            repeat_current_minus_normal_ap50_pp=100*(a['metrics'][K50]-report['models'][name]['test'][K50]),
            interpretation='Same trained temporal weights, all channels current frame: evidence of reliance on temporal input, also an out-of-distribution intervention; not an independently trained ablation.')
    sequences = sorted({r['seq'] for r in index['test'].values()})
    boot_names = ['yolo11n']+MAIN
    prep = {n: prepare_boot(OUT/'evaluations/test'/n, index['test'], sequences) for n in boot_names}
    for name in boot_names:
        assert abs(weighted_ap50(prep[name], np.ones(len(sequences), dtype=int))-report['models'][name]['test'][K50]) < 1e-9, name
    rng = np.random.default_rng(20261017)
    n_boot = 400
    draws = np.zeros((n_boot, len(boot_names)))
    for b in range(n_boot):
        weights = np.bincount(rng.integers(len(sequences), size=len(sequences)), minlength=len(sequences))
        for j, name in enumerate(boot_names):
            draws[b, j] = weighted_ap50(prep[name], weights)
    contrasts = [(n, 'yolo11n') for n in MAIN]+[('C2_temporal_aat_sequence', 'C1_temporal_uniform')]
    report['bootstrap'] = dict(n_sequences=len(sequences), replicates=n_boot, seed=20261017,
        method='Paired resampling of entire held-out sequences with replacement; recompute pooled AP50 with fixed confidence tie order.',
        contrasts={}, caveat='95% percentile intervals are marginal descriptive intervals, not simultaneous or training-seed confidence intervals; no p-values or significance declarations.')
    for left, right in contrasts:
        delta = 100*(draws[:, boot_names.index(left)]-draws[:, boot_names.index(right)])
        report['bootstrap']['contrasts'][left+'__minus__'+right] = dict(
            observed_difference_pp=100*(report['models'][left]['test'][K50]-report['models'][right]['test'][K50]),
            ci95_pp=np.percentile(delta, [2.5, 97.5]).tolist())
    (OUT/'comparison.json').write_text(json.dumps(report, indent=2))
    with (OUT/'comparison.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'epochs', 'val_AP50', 'test_AP50', 'test_AP5095', 'test_delta_AP50_pp', 'test_BG_FP025', 'test_hit025', 'test_TC_AP50', 'test_SV_AP50', 'test_SIZE1_AP50'])
        w.writeheader()
        for n, r in report['models'].items():
            w.writerow(dict(model=n, epochs=r['epochs'], val_AP50=r['val'][K50], test_AP50=r['test'][K50],
                test_AP5095=r['test'][K95], test_delta_AP50_pp=r['test_delta_ap50_pp'],
                test_BG_FP025=r['test_background_fp025']['background_frame_fp_rate'],
                test_hit025=r['test_background_fp025']['positive_frame_hit_rate_iou50'],
                test_TC_AP50=r['test_groups']['TC'][K50], test_SV_AP50=r['test_groups']['SV'][K50],
                test_SIZE1_AP50=r['test_groups']['SIZE_CODE_1_POS_ONLY'][K50]))
    print('ANALYSIS_COMPLETE', json.dumps({n: dict(AP50=r['test'][K50], AP95=r['test'][K95], delta_pp=r['test_delta_ap50_pp'], FP025=r['test_background_fp025']['background_frame_fp_rate']) for n,r in report['models'].items()}), flush=True)


if __name__ == '__main__':
    main()
