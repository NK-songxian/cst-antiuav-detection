"""Audit existing temporal materializations and set up isolated, C1-aligned experiments."""
import argparse,collections,hashlib,json,sys,zipfile
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np
import yaml

ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
OLD=ROOT/'experiments/tcs_ablation_20261017'
AAT=ROOT/'experiments/aat_temporal_100e_20261017'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);a=ap.parse_args()
    a.study.mkdir(exist_ok=False)
    with zipfile.ZipFile(OLD/'relaunch_code.zip') as z:z.extractall(a.study)
    index=json.loads((ROOT/'data/yolo/attribute_index.json').read_text());cv2.setNumThreads(1)
    seq={k:{r['seq'] for r in v.values()} for k,v in index.items()}
    assert not seq['train']&seq['val'] and not seq['train']&seq['test'] and not seq['val']&seq['test']
    reports={};steps=collections.Counter()
    for split in ['train','val']:
        groups=collections.defaultdict(list)
        for r in index[split].values():groups[r['seq']].append(r['frame'])
        for times in groups.values():steps.update(np.diff(sorted(times)).tolist())
    for variant,far in [('far15',15),('far20',20)]:
        d=OLD/'data'/variant
        for split in ['train','val']:
            idx=index[split];hist=json.loads((d/f'history_{split}.json').read_text())
            lookup=collections.defaultdict(dict)
            for stem,r in idx.items():lookup[r['seq']][r['frame']]=stem
            listed=Path(d/f'temporal_{split}.txt').read_text().splitlines()
            assert len(listed)==len(idx) and {Path(p).stem for p in listed}==set(idx)
            repeated=0
            for stem,r in idx.items():
                near=lookup[r['seq']].get(r['frame']-5,stem)
                distant=lookup[r['seq']].get(r['frame']-far,near)
                assert hist[stem]==[distant,near,stem],stem
                for prev in hist[stem]:assert idx[prev]['seq']==r['seq'] and idx[prev]['frame']<=r['frame']
                label=d/split/'labels'/(stem+'.txt')
                assert label.resolve()==(ROOT/'data/yolo'/split/'labels'/(stem+'.txt')).resolve()
                assert (d/split/'images'/(stem+'.png')).is_file()
                repeated+=len(set(hist[stem]))<3
            probes=sorted(idx)[::max(1,len(idx)//120)]
            for stem in probes:
                saved=cv2.imread(str(d/split/'images'/(stem+'.png')))
                assert saved is not None
                for ch,previous in enumerate(hist[stem]):
                    raw=cv2.imread(str(ROOT/'data/yolo'/split/'images'/(previous+'.jpg')),cv2.IMREAD_GRAYSCALE)
                    assert np.array_equal(saved[:,:,ch],raw),stem
            reports[variant+'_'+split]=dict(frames=len(idx),history_sha256=sha(d/f'history_{split}.json'),
                list_sha256=sha(d/f'temporal_{split}.txt'),repeated_boundary_frames=repeated,
                checked_current_labels=len(idx),verified_lossless_images=len(probes),future_frames_used=False)
        cfg=dict(path=str(d),train=str(d/'temporal_train.txt'),val=str(d/'temporal_val.txt'),nc=1,names=['uav'])
        (a.study/f'cst_{variant}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    assert steps[5]>0 and steps[1]==0,'Recheck the premise about raw frame sampling stride'
    (a.study/'data_audit.json').write_text(json.dumps(dict(status='PASS',split_sequences_disjoint=True,
        raw_frame_step_histogram=dict(steps),variants=reports,interpretation='near=5 fixed, far=15/20; not a delta1/delta10 experiment'),indent=2))
    common=json.loads((AAT/'common_args.json').read_text());common.update(project=str(ROOT/'runs/tcs_ablation_relaunch_20261017'),workers=2)
    (a.study/'common_args.json').write_text(json.dumps(common,indent=2))
    matrix=[]
    for name,w,data,wave,channels,temporal in [
        ('tcs_yolo8n','yolov8n.pt',AAT/'cst_temporal.yaml',1,[0,-5,-10],True),
        ('tcs_far15_yolo11n','yolo11n.pt',a.study/'cst_far15.yaml',1,[0,-5,-15],True),
        ('tcs_yolo11s','yolo11s.pt',AAT/'cst_temporal.yaml',2,[0,-5,-10],True),
        ('tcs_far20_yolo11n','yolo11n.pt',a.study/'cst_far20.yaml',2,[0,-5,-20],True),
        ('single_yolo11s','yolo11s.pt',ROOT/'experiments/baselines_100e_20261017/cst_original.yaml',3,[0,0,0],False)]:
        weight=ROOT/w;assert weight.is_file()
        matrix.append(dict(name=name,weights=str(weight),sha256=sha(weight),data=str(data),wave=wave,
            temporal_channels=channels,temporal=temporal,sampling='uniform'))
    (a.study/'matrix.json').write_text(json.dumps(matrix,indent=2))
    (a.study/'scope.json').write_text(json.dumps(dict(created_at=datetime.now(timezone.utc).isoformat(),
        authorization='User requested cleanup and rerun of CST teacher-recommended parallel queue ablations.',
        primary='val mAP50; weights selected by native val AP50-95',test_selection=False,
        runtime_changes=dict(train_workers=2,prefetch_factor=1,pin_memory=False,start_method='forkserver',validation_batch_cap=128),
        justification='Prevent CUDA handles inherited by dead workers and RAM/validation peaks; train batch256 unchanged.',
        additional_control='YOLO11s single-frame matched baseline needed to distinguish model scaling from temporal benefit.'),indent=2))
    print(json.dumps({'status':'DATA_AUDIT_PASS','matrix':[j['name'] for j in matrix],'steps':dict(steps)}),flush=True)


if __name__=='__main__':main()
