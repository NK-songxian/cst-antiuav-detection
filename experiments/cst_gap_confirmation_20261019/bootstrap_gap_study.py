"""Remote preflight + data preparation. No training until every identity check passes."""
import concurrent.futures, hashlib, json, shutil, sys
from datetime import datetime,timezone
from pathlib import Path
import cv2,numpy as np,yaml,torch,ultralytics
from gap_protocol import build_history

ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
HERE=Path(__file__).resolve().parent
OLD=ROOT/'experiments/cst_paper_completion_20261018'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,d):p.write_text(json.dumps(d,indent=2))

def main():
    assert not (HERE/'contract.json').exists(),'Refuse to overwrite frozen study'
    expected=json.loads((HERE/'reference_contract.json').read_text())
    reference_common=json.loads((HERE/'reference_common_args.json').read_text())
    assert sha(OLD/'common_args.json')==sha(HERE/'reference_common_args.json')
    assert sha(OLD/'contract.json')==sha(HERE/'reference_contract.json'),'baseline contract changed'
    for path,value in expected['code_sha256'].items():assert sha(path)==value,path
    assert sha(ROOT/'data/yolo/attribute_index.json')==expected['index_sha256']
    assert sha(Path(expected['models'][1]['data']))==expected['data_sha256']
    weights=ROOT/'yolo11n.pt'
    if not weights.exists():weights=ROOT/'weights/yolo11n.pt'
    assert sha(weights)=='0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'
    assert torch.__version__==expected['environment']['torch']
    assert ultralytics.__version__==expected['environment']['ultralytics']
    assert sys.version==expected['environment']['python']
    reference_metrics=json.loads((HERE/'reference_completion_metrics.json').read_text())
    ref_rows=reference_metrics['rows']
    if isinstance(ref_rows,dict):ref_rows=list(ref_rows.values())
    reuse=[];reference_audits=json.loads((HERE/'reference_audits.json').read_text())
    for seed in [0,1,2]:
        name=f'N_tcs_seed{seed}'
        run=ROOT/'runs/cst_paper_completion_20261018'/name
        audit=json.loads((run.parent/'test_audit'/name/'audit.json').read_text())
        assert sha(run.parent/'test_audit'/name/'audit.json')==reference_audits[name]['audit_sha256']
        assert audit['weights_sha256']==reference_audits[name]['weights_sha256']
        assert sha(run/'weights/best.pt')==audit['weights_sha256']
        args=yaml.safe_load((run/'args.yaml').read_text())
        assert args['seed']==seed and args['batch']==256 and args['epochs']==100 and args['patience']==20
        assert args['data']==next(x for x in expected['models'] if x['name']==name)['data']
        for key,value in reference_common.items():
            if key not in ('data','seed','name','workers'):
                assert args[key]==value,(name,key,args[key],value)
        reuse.append(dict(name=name,weights_sha256=sha(run/'weights/best.pt'),audit_sha256=sha(run.parent/'test_audit'/name/'audit.json'),args_sha256=sha(run/'args.yaml')))
    olddata=ROOT/'experiments/aat_temporal_100e_20261017'
    ready=json.loads((HERE/'reference_temporal_data_ready.json').read_text())
    assert sha(olddata/'temporal_data_ready.json')==sha(HERE/'reference_temporal_data_ready.json')
    for split in ['train','val']:
        hashes=olddata/f'temporal_{split}_image_hashes.txt'
        assert sha(hashes)==ready['splits'][split]['image_hash_list_sha256']
        for line in hashes.read_text().splitlines():
            stem,h=line.split();assert sha(olddata/'data'/split/'images'/(stem+'.png'))==h,stem
        print('BASELINE_PNG_HASH_VERIFIED',split,flush=True)
    index=json.loads((ROOT/'data/yolo/attribute_index.json').read_text())
    seqs={s:{r['seq'] for r in records.values()} for s,records in index.items()}
    assert not seqs['train']&seqs['val'] and not seqs['train']&seqs['test'] and not seqs['val']&seqs['test']
    assert {s:len(index[s]) for s in index}==dict(train=25612,val=9066,test=14446)
    # Do not patch the baseline modules or global Python environment.
    for name in ['matched_runtime.py','train_matched.py']:
        shutil.copy2(OLD/name,HERE/name)
        assert sha(OLD/name)==sha(HERE/name)
    common=json.loads((OLD/'common_args.json').read_text())
    assert common['batch']==256 and common['epochs']==100 and common['patience']==20 and common['imgsz']==640
    common['project']=str(ROOT/'runs/cst_gap_confirmation_20261019')
    dump(HERE/'common_args.json',common)
    cv2.setNumThreads(1)
    reports={};jobs=[]
    # Reject insufficient disk space before creating approximately 100k PNGs.
    source_total=sum(p.stat().st_size for p in (ROOT/'experiments/aat_temporal_100e_20261017/data').rglob('*.png'))
    reserve=max(25*1024**3,2.6*source_total)
    assert shutil.disk_usage(ROOT).free>reserve,('insufficient disk',reserve)
    for far in [15,20]:
        data=HERE/f'gap{far}'
        reports[str(far)]={}
        for split,idx in index.items():
            source=ROOT/'data/yolo'/split
            dest=data/split;(dest/'images').mkdir(parents=True,exist_ok=True);(dest/'labels').mkdir(exist_ok=True)
            history=build_history(idx,far);stems=sorted(idx)
            def make(stem):
                triplet=history[stem]
                ims={s:cv2.imread(str(source/'images'/(s+'.jpg')),cv2.IMREAD_GRAYSCALE) for s in set(triplet)}
                assert all(x is not None for x in ims.values()),stem
                assert len({x.shape for x in ims.values()})==1
                arr=cv2.merge([ims[s] for s in triplet]);out=dest/'images'/(stem+'.png')
                assert not out.exists(),out
                assert cv2.imwrite(str(out),arr,[cv2.IMWRITE_PNG_COMPRESSION,1])
                label=dest/'labels'/(stem+'.txt');label.symlink_to(source/'labels'/(stem+'.txt'))
                assert label.read_bytes()==(source/'labels'/(stem+'.txt')).read_bytes()
                return str(out),sha(out)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:outputs=list(pool.map(make,stems))
            (data/f'{split}.txt').write_text(''.join(p+'\n' for p,h in outputs))
            (data/f'{split}_hashes.txt').write_text(''.join(s+' '+h+'\n' for s,(p,h) in zip(stems,outputs)))
            dump(data/f'{split}_history.json',history)
            # Lossless numerical probes include sequence starts and deterministic interior frames.
            probes=set(stems[::max(1,len(stems)//200)])
            for seq in seqs[split]:probes.add(min((s for s in stems if idx[s]['seq']==seq),key=lambda s:idx[s]['frame']))
            for stem in probes:
                arr=cv2.imread(str(dest/'images'/(stem+'.png')))
                for channel,s in enumerate(history[stem]):
                    assert np.array_equal(arr[:,:,channel],cv2.imread(str(source/'images'/(s+'.jpg')),cv2.IMREAD_GRAYSCALE))
            reports[str(far)][split]=dict(frames=len(stems),sequences=len(seqs[split]),history_sha256=sha(data/f'{split}_history.json'),images_sha256=sha(data/f'{split}_hashes.txt'),list_sha256=sha(data/f'{split}.txt'),probes=len(probes))
            print('DATA_READY',far,split,len(stems),flush=True)
        yp=data/'data.yaml';yp.write_text(yaml.safe_dump(dict(path=str(data),train=str(data/'train.txt'),val=str(data/'val.txt'),test=str(data/'test.txt'),nc=1,names=['uav']),sort_keys=False))
        template=next(x for x in expected['models'] if x['name']=='N_tcs_seed0')
        for seed in [0,1,2]:
            j=dict(template);j.update(name=f'N_gap{far}_seed{seed}',seed=seed,data=str(yp),weights=str(weights),far=far,lane='A' if far==15 else 'B',priority=seed);jobs.append(j)
    codes=dict(expected['code_sha256'])
    for name in ['matched_runtime.py','train_matched.py','gap_protocol.py','bootstrap_gap_study.py','evaluate_gap.py','queue_gap.py']:
        codes[str(HERE/name)]=sha(HERE/name)
    contract=dict(frozen_at=now(),authorization='2026-10-19 user: if supplemental experiments needed arrange on AutoDL, two concurrent jobs and two queues',models=jobs,code_sha256=codes,reference_reuse=reuse,reference_contract_sha256=sha(OLD/'contract.json'),data_reports=reports,common_args_sha256=sha(HERE/'common_args.json'),protocol_sha256=sha(HERE/'experiment_plan.json'),environment=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,python=sys.version,numpy=np.__version__),max_parallel=2,stopping='100 epochs, native patience20, no retry/retune on test; any failure prevents further launches; never reduce batch',selection='native best.pt by validation AP50-95; test only for reporting')
    dump(HERE/'contract.json',contract);print('PREFLIGHT_PASS_SIX_NEW_RUNS',flush=True)

if __name__=='__main__':main()
