"""Two named FIFO lanes. At most two owned GPU jobs; stop new launches after failure."""
import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def room():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    free=int(raw.strip().splitlines()[0])*1024**2
    cg=Path('/sys/fs/cgroup')
    if (cg/'memory.max').exists() and (cg/'memory.max').read_text().strip()!='max':
        limit=int((cg/'memory.max').read_text());used=int((cg/'memory.current').read_text())
        stat=dict(line.split() for line in (cg/'memory.stat').read_text().splitlines());ram=limit-used+int(stat.get('inactive_file',0))
    else:
        ram=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    return free,ram
def main():
    q=HERE/'queue';q.mkdir(exist_ok=True)
    lock=(q/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=json.loads((HERE/'contract.json').read_text());assert len(c['models'])==6 and c['max_parallel']==2
    for path,h in c['code_sha256'].items():assert sha(path)==h,path
    state=dict(scheduler_pid=os.getpid(),started_at=now(),status='running',max_parallel=2,contract_sha256=sha(HERE/'contract.json'),jobs=[dict(j,state='pending',stage='train') for j in c['models']])
    (q/'logs').mkdir(exist_ok=True);active={};halt=False
    def save():
        state['updated_at']=now();state['active_processes']=len(active)
        p=q/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(q/'status.json')
    while True:
        for lane,(proc,job,log) in list(active.items()):
            rc=proc.poll()
            if rc is None:continue
            log.close();del active[lane]
            job['returncode']=rc;job['last_stage_ended_at']=now()
            if rc:
                job['state']='failed';halt=True
            elif job['stage']=='train':job.update(stage='val',state='pending')
            elif job['stage']=='val':job.update(stage='test',state='pending')
            else:job.update(state='completed',ended_at=now())
        if not halt:
            for lane in ['A','B']:
                if lane in active:continue
                pending=[x for x in state['jobs'] if x['lane']==lane and x['state']=='pending']
                if not pending:continue
                job=min(pending,key=lambda x:x['seed']);free,ram=room()
                # Reserve full peak of active training, including not-yet-allocated startup memory.
                startup_reserve=sum(max(0,x[1]['gpu_budget_bytes']-gpu_used(x[0].pid)) if x[1]['stage']=='train' else 0 for x in active.values())
                need=job['gpu_budget_bytes'] if job['stage']=='train' else 11*1024**3
                if free-startup_reserve<need+1024**3 or ram<18*1024**3:
                    state['waiting_reason']=dict(gpu_free=free,unallocated_reservation=startup_reserve,ram_available=ram);continue
                script='train_matched.py' if job['stage']=='train' else 'evaluate_gap.py'
                argv=[sys.executable,str(HERE/script),'--study',str(HERE),'--name',job['name']]
                if job['stage']!='train':argv+=['--split',job['stage']]
                env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONHASHSEED=str(job['seed']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='2',YOLO_AUTOINSTALL='false',MPLBACKEND='Agg',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                log=(q/'logs'/f"{job['name']}_{job['stage']}.log").open('w')
                proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=HERE,start_new_session=True)
                job.update(state='running',pid=proc.pid,stage_started_at=now(),argv=argv);active[lane]=(proc,job,log)
                print('LAUNCHED',lane,job['name'],job['stage'],proc.pid,flush=True)
        save()
        if all(x['state']=='completed' for x in state['jobs']):state['status']='completed';save();break
        if halt and not active:state['status']='failed_stop';save();break
        time.sleep(10)
def gpu_used(pid):
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
    for line in result.stdout.splitlines():
        parts=[x.strip() for x in line.split(',')]
        if len(parts)==2 and parts[0]==str(pid) and parts[1].isdigit():return int(parts[1])*1024**2
    return 0
if __name__=='__main__':main()
