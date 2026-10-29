"""Two barrier-synchronized waves; never signal pre-existing processes."""
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
from pathlib import Path
from datetime import datetime,timezone


def now():return datetime.now(timezone.utc).isoformat()


def resources():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    free_gpu=int(raw.splitlines()[0])*1024**2
    cg=Path('/sys/fs/cgroup')
    limit=int((cg/'memory.max').read_text());used=int((cg/'memory.current').read_text())
    stat=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines())
    reclaim=int(stat.get('inactive_file',0))
    # Kernel may reclaim inactive file cache without stopping another process.
    free_ram=max(0,limit-used+reclaim)
    return dict(gpu_free_bytes=free_gpu,ram_available_including_inactive_file=free_ram,
                ram_limit_bytes=limit,ram_used_bytes=used,inactive_file_bytes=reclaim)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);a=ap.parse_args()
    study=a.study.resolve();root=study.parents[1]
    contract=json.loads((study/'contract.json').read_text());common=json.loads((study/'common_args.json').read_text())
    outputs=Path(common['project']);qd=outputs/'queue';qd.mkdir(parents=True,exist_ok=True)
    lock=(study/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.write(str(os.getpid()));lock.flush()
    jobs=[dict(j,state='pending',attempt=0) for j in contract['models']]
    active={};stopping=False;last_launch=0.;last_wait_message=0.;latest_resource={}
    def log(s):
        line=f'[{now()}] {s}';print(line,flush=True)
        with (qd/'queue.log').open('a') as f:f.write(line+'\n')
    def save(status='running'):
        temp=qd/'status.tmp'
        temp.write_text(json.dumps(dict(updated_at=now(),scheduler_pid=os.getpid(),status=status,
            max_parallel=2,jobs=jobs,resources=latest_resource),indent=2));temp.replace(qd/'status.json')
    def stop(sig,frame):
        nonlocal stopping
        stopping=True;log(f'Stop requested {sig}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    log('START CST corrected four-way: batch256/640/100epochs/patience20; preserve VisDrone jobs, max two CST jobs.')
    try:
        while not stopping:
            for pid,(proc,job,stream) in list(active.items()):
                rc=proc.poll()
                if rc is None:continue
                stream.close()
                try:os.killpg(pid,signal.SIGTERM)
                except ProcessLookupError:pass
                mp=study/'jobs'/job['run_name']/'manifest.json'
                m=json.loads(mp.read_text()) if mp.exists() else {}
                rp=outputs/job['run_name'];result=rp/'results.csv'
                ok=rc==0 and m.get('status')=='completed' and result.is_file() and (rp/'weights/best.pt').is_file()
                if ok:
                    rows=list(csv.DictReader(result.open()));r={k.strip():v for k,v in rows[-1].items()} if rows else {}
                    ok=bool(r) and all(math.isfinite(float(r[k])) for k in ['metrics/mAP50(B)','metrics/mAP50-95(B)'])
                job.update(state='completed' if ok else 'failed',exit_code=rc,ended_at=now())
                log(f'EXIT {job["name"]}: rc={rc} verified_complete={ok}')
                del active[pid]
                # Failed conditions remain visible. Do not silently halve batch or retry a changed experiment.
            if any(j['state']=='failed' for j in jobs):
                save('needs_attention')
                if not active:break
                time.sleep(5);continue
            pending=[j for j in jobs if j['state']=='pending']
            unfinished=[j for j in jobs if j['state'] in {'pending','running'}]
            wave=min((j['wave'] for j in unfinished),default=0)
            choices=[j for j in pending if j['wave']==wave]
            latest_resource=resources()
            need=choices[0]['launch_gpu_bytes'] if choices else 0
            enough=latest_resource['gpu_free_bytes']>=need and latest_resource['ram_available_including_inactive_file']>=14*1024**3
            if choices and len(active)<2 and enough and time.monotonic()-last_launch>=35:
                job=choices[0];job.update(attempt=1,run_name=job['name'])
                stream=(qd/(job['name']+'.log')).open('x')
                env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONHASHSEED='0',OMP_NUM_THREADS='2',
                    MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='2',YOLO_AUTOINSTALL='false',
                    MPLBACKEND='Agg',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                proc=subprocess.Popen([sys.executable,str(study/'train_safe.py'),'--study',str(study),'--name',job['name']],
                    cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                job.update(state='running',pid=proc.pid,started_at=now());active[proc.pid]=(proc,job,stream)
                last_launch=time.monotonic();log(f'LAUNCH {job["name"]}: pid={proc.pid}')
            elif choices and len(active)<2 and not enough and time.monotonic()-last_wait_message>120:
                log('WAIT_RESOURCES '+json.dumps(latest_resource));last_wait_message=time.monotonic()
            save()
            if not active and not pending:break
            time.sleep(5)
    finally:
        if stopping:
            for pid,(proc,job,stream) in active.items():
                # Only process groups created by this scheduler are eligible.
                if proc.poll() is None:os.killpg(pid,signal.SIGTERM)
                job.update(state='cancel_requested',ended_at=now());stream.close()
        final='completed' if all(j['state']=='completed' for j in jobs) else ('stopping' if stopping else 'needs_attention')
        save(final);log('QUEUE_END '+final)


if __name__=='__main__':main()
