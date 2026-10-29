"""Freeze paired input ablations and three training seeds; reuse the live 11s job."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,subprocess
import psutil,torch,ultralytics
ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
HERE=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert json.loads((HERE/'preflight.json').read_text())['status']=='PASS'
    previous=ROOT/'experiments/tcs11s_recovery_20261018'
    common=json.loads((previous/'common_args.json').read_text());common['project']=str(ROOT/'runs/cst_paper_completion_20261018')
    common.update(workers=8,seed=0)
    (HERE/'common_args.json').write_text(json.dumps(common,indent=2))
    qs=ROOT/'runs/tcs11s_recovery_20261018/queue/status.json';q=json.loads(qs.read_text())
    assert q['status']=='running'
    live=next(j for j in q['jobs'] if j['name']=='tcs_yolo11s');assert live['state']=='running'
    parent=psutil.Process(q['scheduler_pid']);child=psutil.Process(live['pid'])
    data=ROOT/'experiments/aat_temporal_100e_20261017/cst_temporal.yaml'
    rows=[]
    def add(name,arch,mode,seed,priority,workers=8,**extra):
        w=ROOT/(arch+'.pt')
        rows.append(dict(name=name,arch=arch,input_mode=mode,seed=seed,priority=priority,workers=workers,data=str(data),
            weights=str(w),weights_sha256=sha(w),gpu_budget_bytes=int((60 if arch=='yolo11s' else 33.5)*1024**3),
            allocator_cap_bytes=int((75 if arch=='yolo11s' else 35)*1024**3),**extra))
    add('N_current_seed0','yolo11n','current',0,0)
    add('N_tcs_seed0','yolo11n','temporal',0,1)
    add('S_current_seed0','yolo11s','current',0,2,workers=2)
    add('S_tcs_seed0_adopted','yolo11s','temporal',0,2,workers=2,adopted=True,
        trained_weights=str(ROOT/'runs/tcs11s_recovery_20261018/tcs_yolo11s/weights/best.pt'))
    for seed in [1,2]:
        add(f'N_current_seed{seed}','yolo11n','current',seed,2+seed*2)
        add(f'N_tcs_seed{seed}','yolo11n','temporal',seed,3+seed*2)
    deps=[HERE/x for x in ['matched_runtime.py','train_matched.py','evaluate_matched.py','queue_matched.py','test_matched_runtime.py']]
    deps += [ROOT/'experiments/aat_temporal_100e_20261017'/x for x in ['train_innovation.py','sampling.py']]
    deps += [ROOT/'experiments/baselines_100e_20261017/audit_validation.py']
    contract=dict(frozen_at=datetime.now(timezone.utc).isoformat(),authorization='User asked to revise the outline and immediately complete missing experiments with large batch.',
        primary='paired seed-wise test AP50 difference; native checkpoint selected on val AP50-95',
        scientific_comparison='same temporal PNG data and current labels, same uniform sampler and shared brightness; only input mode differs within each paired seed',
        training=dict(batch=256,imgsz=640,epochs=100,patience=20,seeds=[0,1,2],no_test_selection=True,
            workers_nano=8,workers_small=2,val_loader_workers=4,val_batch=128),
        models=rows,code_sha256={str(p):sha(p) for p in deps},index_sha256=sha(ROOT/'data/yolo/attribute_index.json'),
        data_sha256=sha(data),preflight_sha256=sha(HERE/'preflight.json'),
        adopted_training=dict(pid=child.pid,create_time=child.create_time(),
            manifest=str(previous/'jobs/tcs_yolo11s/manifest.json'),output=str(ROOT/'runs/tcs11s_recovery_20261018/tcs_yolo11s')),
        transferred_scheduler=dict(pid=parent.pid,create_time=parent.create_time(),status_path=str(qs)),
        failure_tree=dict(success='Report all three paired seeds, mean and SD, with full attribute/FP results',
            inconclusive='Report spread and limitations; do not select best seed',failure='Preserve failed run and stop new launches; no silent batch/config changes'),
        environment=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,python=sys.version))
    (HERE/'contract.json').write_text(json.dumps(contract,indent=2));(HERE/'matrix.json').write_text(json.dumps(rows,indent=2))
    print('FROZEN',json.dumps({'rows':len(rows),'new_training':7,'adopted_training':1,'batch':256}),flush=True)
if __name__=='__main__':main()
