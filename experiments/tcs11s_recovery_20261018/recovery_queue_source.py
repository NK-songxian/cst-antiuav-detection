"""Complete the two previously authorized 11s controls, in isolation after review."""
import hashlib,json,os,subprocess,sys,time,signal,fcntl,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
OLD=ROOT/'experiments/tcs_ablation_relaunch_20261017'
STUDY=ROOT/'experiments/tcs11s_recovery_20261018'
REVIEW=ROOT.parent/'uav_sclr/experiments/bcg_p2_4way_20261017/completed_review_20261018'
def now():return datetime.now(timezone.utc).isoformat()
def main():
    STUDY.mkdir(exist_ok=False)
    common=json.loads((OLD/'common_args.json').read_text());common['project']=str(ROOT/'runs/tcs11s_recovery_20261018')
    c=json.loads((OLD/'contract.json').read_text())
    c['models']=[j for j in c['models'] if j['name'] in ['tcs_yolo11s','single_yolo11s']]
    for j in c['models']:
        j['allocator_cap_bytes']=80*1024**3;j['launch_gpu_bytes']=84*1024**3
    c['recovery']=dict(source_contract_sha256=hashlib.sha256((OLD/'contract.json').read_bytes()).hexdigest(),
        reason='Previous tcs_yolo11s OOM during concurrent execution; single-frame control was never launched.',
        changes=['isolated output folder','one training process maximum','80GiB allocator cap;84GiB free required'],
        unchanged='model/data/train batch256/imgsz640/epochs100/patience20/seed0/optimizer auto/sampler/augmentations',
        authorization='Complete the user-authorized CST ablation queue; not a new scientific condition.')
    (STUDY/'contract.json').write_text(json.dumps(c,indent=2));(STUDY/'common_args.json').write_text(json.dumps(common,indent=2))
    shutil.copy2(OLD/'train_safe.py',STUDY/'train_safe.py')
    shutil.copy2(__file__,STUDY/'recovery_queue_source.py')
    lock=(STUDY/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    qd=Path(common['project'])/'queue';qd.mkdir(parents=True)
    state=dict(status='waiting_for_evaluation_and_benchmark',scheduler_pid=os.getpid(),max_parallel=1,
        jobs=[dict(name=j['name'],state='pending') for j in c['models']])
    def save():state['updated_at']=now();(qd/'status.json').write_text(json.dumps(state,indent=2))
    save()
    while not (REVIEW/'benchmark_complete.json').exists():time.sleep(10)
    for entry in state['jobs']:
        state['status']='waiting_for_resources';save()
        while True:
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])*1024**2
            if free>=84*1024**3:break
            time.sleep(15)
        env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',PYTHONHASHSEED='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
          OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='2',YOLO_AUTOINSTALL='false',MPLBACKEND='Agg',
          PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (qd/(entry['name']+'.log')).open('x') as stream:
            proc=subprocess.Popen([sys.executable,str(STUDY/'train_safe.py'),'--study',str(STUDY),'--name',entry['name']],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            state['status']='running';entry.update(state='running',pid=proc.pid,started_at=now());save()
            rc=proc.wait()
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        mpath=STUDY/'jobs'/entry['name']/'manifest.json'
        manifest=json.loads(mpath.read_text()) if mpath.exists() else {}
        ok=rc==0 and manifest.get('status')=='completed'
        entry.update(state='completed' if ok else 'failed',exit_code=rc,ended_at=now());save()
        if not ok:state['status']='needs_attention';save();return
    state['status']='completed';save()
if __name__=='__main__':main()
