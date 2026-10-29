"""Collect immutable completed-run evidence and the common validation audit."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY = ROOT / 'experiments/baselines_100e_20261017'
RUNS = ROOT / 'runs/baselines_100e_20261017'
OUT = ROOT / 'experiments/innovation_plan_20261017'


def read_csv(path):
    with path.open() as f:
        return [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]


def main():
    OUT.mkdir(exist_ok=True)
    fixed = json.loads((STUDY / 'fixed_fpr_diagnostic.json').read_text())['results']
    records = []
    for p in sorted((RUNS / 'validation_audit').glob('*/audit.json')):
        audit = json.loads(p.read_text())
        name = audit['name']
        rows = read_csv(Path(audit['weights']).parents[1] / 'results.csv')
        best = max(rows, key=lambda x: float(x['metrics/mAP50-95(B)']))
        g = audit['groups']
        records.append(dict(name=name, legacy=name.endswith('_legacy'), epochs_recorded=len(rows),
                            best_epoch=int(best['epoch']),
                            train_csv_best_ap50=float(best['metrics/mAP50(B)']),
                            train_csv_best_ap5095=float(best['metrics/mAP50-95(B)']),
                            audited_ap50=g['ALL']['metrics/mAP50(B)'],
                            audited_ap5095=g['ALL']['metrics/mAP50-95(B)'],
                            audited_tc_ap50=g['TC']['metrics/mAP50(B)'],
                            audited_size_code1_ap50=g['SIZE_CODE_1_POS_ONLY']['metrics/mAP50(B)'],
                            background_fp_rate_at_conf025=g['NEG_ONLY']['conf_0.25']['background_frame_fp_rate'],
                            positive_hit_rate_at_fpr005=fixed[name]['0.05']['positive_hit_rate_iou50'],
                            audit_path=str(p), audit_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                            weights_sha256=audit['weights_sha256']))
    old = Path(json.loads((ROOT / 'baseline_reset_latest.json').read_text())['old_runs'])
    old_rows = read_csv(old / 'baseline_yolo11n/results.csv')
    new_rows = read_csv(RUNS / 'yolo11n/results.csv')
    keys = ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'train/box_loss', 'train/cls_loss']
    differences = {k: max(abs(float(x[k]) - float(y[k])) for x, y in zip(old_rows, new_rows)) for k in keys}
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  stage='completed_baselines_and_exploratory_validation_analysis',
                  baseline_queue=json.loads((RUNS / 'queue/status.json').read_text()),
                  shared_baseline_epochs=min(len(old_rows), len(new_rows)), max_abs_delta=differences,
                  results=records, limitations=['Single training seed; no significance or SOTA claim.',
                      'Legacy AAT runs were interrupted and had different sampling budgets.',
                      'Audit uses batch=128, half=True; compare audit against audit, not rounded training logs.',
                      'Fixed-FPR thresholds selected on validation for diagnosis; not held-out test results.',
                      'Attribute flags overlap. SIZE is multi-valued, and OC/OV subsets contain very few positives.'])
    (OUT / 'progress_analysis.json').write_text(json.dumps(result, indent=2))
    with (OUT / 'comparison.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    print('SAVED', OUT / 'progress_analysis.json', 'models', len(records))


if __name__ == '__main__':
    main()
