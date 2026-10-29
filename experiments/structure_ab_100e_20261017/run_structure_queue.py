"""Two independent training process groups; real completion, cleanup and one retry."""
import argparse
import csv
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def successful(exit_code, manifest, csv_path, best_path):
    if exit_code != 0 or manifest.get('status') != 'completed' or not best_path.is_file() or not csv_path.is_file():
        return False
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return False
    row = {k.strip(): v for k, v in rows[-1].items()}
    return all(math.isfinite(float(row[k])) for k in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)'])


def clean_group(pgid):
    if pgid == os.getpgrp():
        raise RuntimeError('Refusing to signal scheduler process group')
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', type=Path, required=True)
    args = ap.parse_args()
    study = args.study.resolve()
    contract = json.loads((study / 'contract.json').read_text())
    common = json.loads((study / 'common_args.json').read_text())
    outputs = Path(common['project'])
    queue_dir = outputs / 'queue'
    queue_dir.mkdir(parents=True, exist_ok=True)
    lock = (study / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    jobs = [dict(**m, state='pending', attempt=0) for m in contract['models']]
    active = {}
    log_path = queue_dir / 'queue.log'
    stopping = False

    def log(message):
        line = f'[{now()}] {message}'
        print(line, flush=True)
        with log_path.open('a') as f:
            f.write(line + '\n')

    def save():
        state = dict(updated_at=now(), scheduler_pid=os.getpid(), max_parallel=2,
                     status='stopping' if stopping else 'running', jobs=jobs)
        temp = queue_dir / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2)); temp.replace(queue_dir / 'status.json')

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        log(f'Stop requested: signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    last_launch = 0.
    log('START: tasks A and B, maximum two simultaneous jobs')
    try:
        while not stopping:
            for pid, record in list(active.items()):
                proc, job, stream = record
                rc = proc.poll()
                if rc is None:
                    continue
                stream.close()
                clean_group(pid)
                run_name = job['run_name']
                mp = study / 'jobs' / run_name / 'manifest.json'
                manifest = json.loads(mp.read_text()) if mp.exists() else {}
                ok = successful(rc, manifest, outputs / run_name / 'results.csv', outputs / run_name / 'weights/best.pt')
                job.update(exit_code=rc, ended_at=now(), state='completed' if ok else 'failed')
                job.setdefault('attempt_history', []).append(dict(attempt=job['attempt'], run_name=run_name,
                                                                 exit_code=rc, success=ok))
                if not ok and job['attempt'] < 2:
                    job['state'] = 'pending'
                    log(f"RETRY_QUEUED {job['name']}: attempt {job['attempt']} failed ({rc}); process group cleaned")
                else:
                    log(f"EXIT {job['name']}: code={rc}, completed={ok}")
                del active[pid]
            pending = [j for j in jobs if j['state'] == 'pending']
            # Stagger CUDA initialization and data scans, while keeping two sustained jobs.
            if pending and len(active) < 2 and time.monotonic() - last_launch >= 30:
                job = pending[0]
                job['attempt'] += 1
                run_name = job['name'] if job['attempt'] == 1 else f"{job['name']}_attempt{job['attempt']}"
                stream = (queue_dir / (run_name + '.log')).open('x')
                env = os.environ.copy()
                env.update(PYTHONUNBUFFERED='1', PYTHONHASHSEED='0', OMP_NUM_THREADS='2',
                           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='2',
                           YOLO_AUTOINSTALL='false', MPLBACKEND='Agg', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
                proc = subprocess.Popen([sys.executable, str(study / 'train_structure.py'),
                                         '--study', str(study), '--name', job['name'],
                                         '--attempt', str(job['attempt'])],
                                        cwd=study.parents[1], env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                job.update(state='running', pid=proc.pid, run_name=run_name, started_at=now())
                active[proc.pid] = (proc, job, stream)
                last_launch = time.monotonic()
                log(f"LAUNCH {job['name']} attempt={job['attempt']} pid={proc.pid}")
            save()
            if not active and not any(j['state'] == 'pending' for j in jobs):
                break
            time.sleep(3)
    finally:
        for pid, (proc, job, stream) in active.items():
            clean_group(pid); stream.close(); job.update(state='cancelled', ended_at=now())
        save()
    completed = sum(j['state'] == 'completed' for j in jobs)
    final = dict(updated_at=now(), scheduler_pid=os.getpid(), max_parallel=2,
                 status='completed' if completed == len(jobs) else 'incomplete', jobs=jobs)
    (queue_dir / 'status.json').write_text(json.dumps(final, indent=2))
    log(f'QUEUE_END completed={completed}/{len(jobs)}')


if __name__ == '__main__':
    main()


