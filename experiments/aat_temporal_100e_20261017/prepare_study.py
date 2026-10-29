"""Freeze the queue and create lossless, split-isolated causal context images."""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from sampling import build_history,build_schedule

ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
BASE=ROOT/'experiments/baselines_100e_20261017'
STUDY=ROOT/'experiments/aat_temporal_100e_20261017'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    assert not (STUDY/'contract.json').exists()
    original=json.loads((BASE/'contract.json').read_text())
    common=json.loads((BASE/'common_args.json').read_text())
    common['project']=str(ROOT/'runs/aat_temporal_100e_20261017')
    (STUDY/'common_args.json').write_text(json.dumps(common,indent=2))
    index=json.loads((ROOT/'data/yolo/attribute_index.json').read_text())
    # Reuse the original split inventory and enforce sequence disjointness explicitly.
    sequences={s:{r['seq'] for r in records.values()} for s,records in index.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not sequences[a]&sequences[b],(a,b)
    idx=index['train']
    rows=[]
    for name,mode,temporal,wave in [('A1_aat_global','global',False,1),
            ('A2_aat_sequence','quota',False,1),('C1_temporal_uniform','uniform',True,2),
            ('C2_temporal_aat_sequence','quota',True,2)]:
        rows.append(dict(name=name,sampling=mode,temporal=temporal,wave=wave,
            data=str(STUDY/'cst_temporal.yaml') if temporal else common['data'],
            weights=str(ROOT/'yolo11n.pt'),sha256=sha(ROOT/'yolo11n.pt')))
    previews={}
    for epoch in (0,1,50,99):
        plans={m:build_schedule(idx,m,common['seed'],epoch) for m in ('global','quota','uniform')}
        records={}
        for mode,order in plans.items():
            counts=Counter(order)
            assert len(order)==25612 and max(counts.values())<=2
            assert Counter(s for s in order if not idx[s]['exist'])==Counter(s for s in idx if not idx[s]['exist'])
            if mode=='quota':
                assert Counter(idx[s]['seq'] for s in order)==Counter(r['seq'] for r in idx.values())
            records[mode]=dict(primary_slots=len(order),unique_frames=len(counts),
                negative_primary_slots=sum(not idx[s]['exist'] for s in order),
                tc_positive_slots=sum(idx[s]['exist']==1 and idx[s]['attr'].get('TC')==1 for s in order),
                exact_sequence_counts=Counter(idx[s]['seq'] for s in order)==Counter(r['seq'] for r in idx.values()))
        assert records['global']['tc_positive_slots']==records['quota']['tc_positive_slots']
        previews[epoch]=records
    (STUDY/'sampling_preflight.json').write_text(json.dumps(previews,indent=2))
    contract=dict(models=rows,common_args_sha256=sha(STUDY/'common_args.json'),
        splits=original['splits'],index_sha256=sha(ROOT/'data/yolo/attribute_index.json'),
        environment=original['environment'],runtime=original['runtime'],max_parallel=2,waves=2,
        source_authorization='User 2026-10-17: arrange two concurrent processes and two batches, AAT then other methods',
        frozen_at=datetime.now(timezone.utc).isoformat(),
        question='Can sequence-preserving AAT improve detection without the background tradeoff, and does causal motion context help?',
        primary='val mAP50; planned target +1.0 percentage point over 66.2124, not a promised gain',
        secondary=['val mAP50-95','TC/SV subgroup AP50','background frame FP@0.25','parameters','image-to-box latency including context construction'],
        guardrail='A candidate with BG FP rate > baseline+2 percentage points or recall <-1 point is reported as a tradeoff, not an unqualified improvement',
        stopping='Maximum 100 epochs, patience 20 on framework AP50-95 fitness; preserve both bestAP50.pt and native best.pt',
        failure='If no val AP50 improvement, retain all runs and do not claim the mechanism improves accuracy',
        no_test_selection=True,seed_role='one exploratory seed 0; no significance or SOTA claim from this queue',
        temporal='BGR image file [t-10,t-5,t], RGB network order [t,t-5,t-10]; input images only, current frame labels only, no future frames',
        temporal_augmentation='Shared scalar brightness LUT replaces HSV across time channels; geometric and Mosaic operations remain unchanged',
        sampling_scope='exact primary slots per epoch; secondary Mosaic draws retain the standard framework buffer policy',
        code_sha256={p.name:sha(p) for p in STUDY.glob('*.py')})
    (STUDY/'contract.json').write_text(json.dumps(contract,indent=2))
    data=dict(path=str(STUDY/'data'),train=str(STUDY/'temporal_train.txt'),
              val=str(STUDY/'temporal_val.txt'),nc=1,names=['uav'])
    (STUDY/'cst_temporal.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
    print(json.dumps(dict(status='FROZEN',previews=previews[0],models=rows)),flush=True)


def temporal():
    cv2.setNumThreads(1)
    assert not (STUDY/'temporal_data_ready.json').exists()
    index=json.loads((ROOT/'data/yolo/attribute_index.json').read_text())
    reports={}
    for split in ('train','val'):
        idx=index[split]
        history=build_history(idx)
        current=STUDY/'data'/split
        (current/'images').mkdir(parents=True,exist_ok=True)
        (current/'labels').mkdir(exist_ok=True)
        source=ROOT/'data/yolo'/split
        def make(stem):
            triplet=history[stem]
            ims={s:cv2.imread(str(source/'images'/(s+'.jpg')),cv2.IMREAD_GRAYSCALE) for s in set(triplet)}
            assert all(im is not None for im in ims.values()),stem
            assert len({im.shape for im in ims.values()})==1,stem
            merged=cv2.merge([ims[s] for s in triplet])
            dest=current/'images'/(stem+'.png')
            assert cv2.imwrite(str(dest),merged,[cv2.IMWRITE_PNG_COMPRESSION,1]),dest
            label=current/'labels'/(stem+'.txt')
            if not label.exists():
                label.symlink_to(source/'labels'/(stem+'.txt'))
            return str(dest),sha(dest)
        stems=sorted(idx)
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            outputs=list(pool.map(make,stems))
        (STUDY/f'temporal_{split}.txt').write_text('\n'.join(p for p,h in outputs)+'\n')
        (STUDY/f'history_{split}.json').write_text(json.dumps(history))
        hash_list=''.join(stem+' '+h+'\n' for stem,(p,h) in zip(stems,outputs))
        (STUDY/f'temporal_{split}_image_hashes.txt').write_text(hash_list)
        # Verify a deterministic spread of actual saved images, including every sequence start.
        probes=stems[::max(1,len(stems)//120)]
        errors=[]
        for stem in probes:
            saved=cv2.imread(str(current/'images'/(stem+'.png')))
            for channel,previous in enumerate(history[stem]):
                raw=cv2.imread(str(source/'images'/(previous+'.jpg')),cv2.IMREAD_GRAYSCALE)
                assert np.array_equal(saved[:,:,channel],raw),stem
            assert (current/'labels'/(stem+'.txt')).read_bytes()==(source/'labels'/(stem+'.txt')).read_bytes()
        reports[split]=dict(frames=len(stems),sequence_count=len({r['seq'] for r in idx.values()}),
            history_full=sum(len(set(v))==3 for v in history.values()),
            missing_history_clamped=sum(len(set(v))<3 for v in history.values()),
            image_hash_list_sha256=sha(STUDY/f'temporal_{split}_image_hashes.txt'),
            history_sha256=sha(STUDY/f'history_{split}.json'),verified_lossless_probes=len(probes))
        print('TEMPORAL_READY',split,json.dumps(reports[split]),flush=True)
    result=dict(status='PASS',splits=reports,future_frames_used=False,history_uses_labels=False,
                test_images_processed=False,created_at=datetime.now(timezone.utc).isoformat())
    (STUDY/'temporal_data_ready.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--temporal',action='store_true');a=p.parse_args()
    temporal() if a.temporal else freeze()
