"""Two read-only validation workers; no new training jobs."""
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY = ROOT / 'experiments/baselines_100e_20261017'


def run(job):
    name, weights = job
    log_dir = STUDY / 'validation_logs'
    log_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               YOLO_AUTOINSTALL='false', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    with (log_dir / (name + '.log')).open('x') as stream:
        p = subprocess.run([sys.executable, str(STUDY / 'audit_validation.py'),
                            '--name', name, '--weights', str(weights)],
                           cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return dict(name=name, exit_code=p.returncode)


def main():
    old = Path(json.loads((ROOT / 'baseline_reset_latest.json').read_text())['old_runs'])
    jobs = [(n, ROOT / 'runs/baselines_100e_20261017' / n / 'weights/best.pt')
            for n in ['yolo11n', 'yolov8n', 'yolo26n', 'yolov5nu']]
    jobs += [(n, old / folder / 'weights/best.pt') for n, folder in [
        ('AAT_v1_legacy', 'AAT_v1_yolo11n'), ('AAT_v2_legacy', 'AAT_v2_yolo11n')]]
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for f in concurrent.futures.as_completed(futures):
            records.append(f.result())
            (STUDY / 'validation_status.json').write_text(json.dumps(records, indent=2))
            print(records[-1], flush=True)


if __name__ == '__main__':
    main()
