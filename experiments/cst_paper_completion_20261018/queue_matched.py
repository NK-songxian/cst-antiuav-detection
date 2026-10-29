"""Two processes max, budget reservations cover warming processes; adopt live 11s without stopping it."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
import psutil
def now():return datetime.now(timezone.utc).isoformat()
GiB=1024**3
def resources():
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])*1024**2
    used={}
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    for line in raw.splitlines():
        try:p,m=line.split(',');used[int(p)]=int(m)*1024**2
        except ValueError:pass
    cg=Path('/sys/fs/cgroup');s=dict(x.split() for x in (cg/'memory.stat').read_text().splitlines())
    ram=int((cg/'memory.max').read_text())-int((cg/'memory.current').read_text())+int(s.get('inactive_file',0))
    return free,used,ram
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);a=ap.parse_args();s=a.study
    c=json.loads((s/'contract.json').read_text());common=json.loads((s/'common_args.json').read_text())
    q=Path(common['project'])/'queue';q.mkdir(parents=True,exist_ok=True)
    lock=(s/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    adopted=dict(c['adopted_training'],name='adopted_tcs_yolo11s',state='running',stage='adopted_training')
    p=psutil.Process(adopted['pid']);assert abs(p.create_time()-adopted['create_time'])<.01
    old=psutil.Process(c['transferred_scheduler']['pid'])
    assert abs(old.create_time()-c['transferred_scheduler']['create_time'])<.01
    assert any('recover_11s_queue.py' in x for x in old.cmdline())
    # The old scheduler starts each training child in its own process group. Signal parent only.
    assert os.getpgid(adopted['pid'])==adopted['pid']
    (s/'scheduler_transfer.json').write_text(json.dumps(dict(at=now(),training_pid_preserved=adopted['pid'],
        scheduler_pid_stopped=old.pid,reason='Replace pending native-JPG control with matched-PNG control and add paired seeds.'),indent=2))
    old.terminate();old.wait(timeout=10);assert psutil.pid_exists(adopted['pid'])
    old_status=Path(c['transferred_scheduler']['status_path']);old_state=json.loads(old_status.read_text())
    old_state.update(status='transferred_to_matched_queue',new_status_path=str(q/'status.json'))
    old_status.write_text(json.dumps(old_state,indent=2))
    jobs=[dict(j,state='waiting_for_adopted' if j.get('adopted') else 'pending',stage='val' if j.get('adopted') else 'train') for j in c['models']]
    active={adopted['pid']:dict(job=adopted,proc=None,stream=None,budget=60*GiB)}
    def save(status='running'):
        payload=dict(status=status,updated_at=now(),scheduler_pid=os.getpid(),max_parallel=2,adopted=adopted,jobs=jobs,
            resource_policy='Reserve full active budgets including processes still warming; global headroom0.5GiB; no auto batch reduction')
        temp=q/'status.tmp';temp.write_text(json.dumps(payload,indent=2));temp.replace(q/'status.json')
    def log(msg):
        with (q/'queue.log').open('a') as f:f.write(now()+' '+msg+'\n')
        print(msg,flush=True)
    save();log('ADOPT training pid '+str(adopted['pid'])+'; old pending single11s replaced, no training stopped')
    failed=False
    while True:
        for pid,row in list(active.items()):
            job=row['job'];proc=row['proc']
            if proc is None:
                mp=Path(adopted['manifest']);m=json.loads(mp.read_text())
                alive=psutil.pid_exists(pid) and abs(psutil.Process(pid).create_time()-adopted['create_time'])<.01
                if alive and m.get('status') not in ['completed','failed']:continue
                ok=m.get('status')=='completed';job['state']='completed' if ok else 'failed'
                if not ok:failed=True
                else:
                    for waiting in jobs:
                        if waiting['state']=='waiting_for_adopted':waiting['state']='pending'
                del active[pid];log('ADOPTED_FINISHED '+job['state']);continue
            rc=proc.poll()
            if rc is None:continue
            row['stream'].close()
            try:os.killpg(pid,signal.SIGTERM)
            except ProcessLookupError:pass
            ok=rc==0
            if job['stage']=='train':
                mp=s/'jobs'/job['name']/'manifest.json';m=json.loads(mp.read_text()) if mp.exists() else {}
                ok=ok and m.get('status')=='trained'
                job.update(state='pending' if ok else 'failed',stage='val' if ok else 'train',train_exit_code=rc)
            elif job['stage']=='val':job.update(state='pending' if ok else 'failed',stage='test' if ok else 'val')
            else:job.update(state='completed' if ok else 'failed',ended_at=now())
            if not ok:failed=True
            del active[pid];log('EXIT '+job['name']+' '+str(rc)+' next='+job['stage'])
        pending=[j for j in jobs if j['state']=='pending']
        if failed and not active:save('needs_attention');break
        if not active and not pending:save('completed');break
        if not failed and len(active)<2:
            free,used,ram=resources()
            reservation=sum(max(0,row['budget']-used.get(pid,0)) for pid,row in active.items())
            available=free-reservation-.5*GiB
            # Audit a completed model before starting another long training run.
            choices=sorted(pending,key=lambda j:(0 if j['stage']!='train' else 1,j['priority']))
            for job in choices:
                budget=(10*GiB if job['stage']!='train' else job['gpu_budget_bytes'])
                if available<budget or ram<18*GiB:continue
                stage=job['stage'];stream=(q/(job['name']+'_'+stage+'.log')).open('x')
                script='train_matched.py' if stage=='train' else 'evaluate_matched.py'
                cmd=[sys.executable,str(s/script),'--study',str(s),'--name',job['name']]
                if stage!='train':cmd+=['--split',stage]
                env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONHASHSEED=str(job['seed']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='2',YOLO_AUTOINSTALL='false',MPLBACKEND='Agg',
                    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                proc=subprocess.Popen(cmd,cwd=s,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                job.update(state='running',pid=proc.pid,started_at=job.get('started_at',now()),current_stage_started_at=now())
                active[proc.pid]=dict(job=job,proc=proc,stream=stream,budget=budget)
                log('LAUNCH '+job['name']+' '+stage+' pid='+str(proc.pid));break
        save('needs_attention' if failed else 'running');time.sleep(5)
if __name__=='__main__':main()
