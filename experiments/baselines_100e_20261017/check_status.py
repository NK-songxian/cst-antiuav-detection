"""Read-only compact snapshot of queue, epoch metrics and resource usage."""
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY = ROOT / 'experiments/baselines_100e_20261017'
OUT = ROOT / 'runs/baselines_100e_20261017'


def main():
    state = json.loads((OUT / 'queue/status.json').read_text())
    result = dict(queue_status=state['status'], updated_at=state['updated_at'],
                  scheduler_pid=state['scheduler_pid'], jobs=[])
    for job in state['jobs']:
        name = job.get('run_name', job['name'])
        entry = {k: job.get(k) for k in ['name', 'state', 'pid', 'attempt', 'exit_code']}
        p = OUT / name / 'results.csv'
        if p.exists():
            rows = list(csv.DictReader(p.open()))
            entry['epochs_recorded'] = len(rows)
            entry['last_epoch'] = rows[-1] if rows else None
        m = STUDY / 'jobs' / name / 'manifest.json'
        if m.exists():
            manifest = json.loads(m.read_text())
            entry['training'] = {k: manifest.get(k) for k in [
                'status', 'actual_model_parameters', 'actual_optimizer',
                'actual_train_images', 'actual_val_images', 'error']}
        p = OUT / 'queue' / (name + '.log')
        if p.exists():
            text = p.read_text(errors='replace')
            entry['log_bytes'] = p.stat().st_size
            entry['log_tail'] = text[-1200:]
        result['jobs'].append(entry)
    for key, command in [
        ('gpu', ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader']),
        ('gpu_processes', ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'])]:
        result[key] = subprocess.check_output(command, text=True).strip()
    for key in ['memory.current', 'memory.max', 'memory.events']:
        p = Path('/sys/fs/cgroup') / key
        if p.exists():
            result[key] = p.read_text().strip()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
