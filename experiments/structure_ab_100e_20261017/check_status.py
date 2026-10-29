"""Read-only status for the two named structure experiments."""
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY = ROOT / 'experiments/structure_ab_100e_20261017'
OUT = ROOT / 'runs/structure_ab_100e_20261017'


def main():
    p = OUT / 'queue/status.json'
    state = json.loads(p.read_text()) if p.exists() else {'status': 'not_launched', 'jobs': []}
    result = {'queue_status': state['status'], 'scheduler_pid': state.get('scheduler_pid'), 'jobs': []}
    for j in state['jobs']:
        name = j.get('run_name', j['name'])
        rec = {k: j.get(k) for k in ['name', 'state', 'pid', 'attempt']}
        p = STUDY / 'jobs' / name / 'manifest.json'
        if p.exists():
            a = json.loads(p.read_text())
            rec['training'] = {k: a.get(k) for k in ['status', 'actual_model_parameters', 'actual_optimizer',
                             'actual_train_images', 'actual_val_images', 'strides', 'contrast_bn_abs_mean', 'error']}
        p = OUT / name / 'results.csv'
        if p.exists():
            a = list(csv.DictReader(p.open()))
            rec['epochs'] = len(a)
            rec['last_metrics'] = a[-1] if a else None
        p = OUT / 'queue' / (name + '.log')
        if p.exists():
            rec['log_tail'] = p.read_text(errors='replace')[-950:]
        result['jobs'].append(rec)
    result['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader'], text=True).strip()
    result['gpu_pids'] = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'], text=True).strip()
    for n in ['memory.current', 'memory.events']:
        result[n] = (Path('/sys/fs/cgroup') / n).read_text().strip()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
