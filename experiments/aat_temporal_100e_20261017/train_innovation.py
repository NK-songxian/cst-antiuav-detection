"""Four controlled YOLO11n experiments; no changes to the network or detection loss."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from ultralytics import YOLO
from ultralytics.data import build
from ultralytics.data.build import seed_worker
from ultralytics.models.yolo.detect import DetectionTrainer
from sampling import build_schedule

ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY=ROOT/'experiments/aat_temporal_100e_20261017'
ACTIVE_JOB=None


def now():
    return datetime.now(timezone.utc).isoformat()


class SharedBrightness:
    """Same V gain draw as RandomHSV; one LUT across time, no time-channel mixing."""
    def __init__(self,hgain=0.015,sgain=0.7,vgain=0.4):
        self.vgain=vgain

    def __call__(self,labels):
        r=np.random.uniform(-1,1,3)
        lut=np.clip(np.arange(256,dtype=np.float64)*(1+r[2]*self.vgain),0,255).astype(np.uint8)
        labels['img']=cv2.LUT(labels['img'],lut)
        return labels


class EpochSampler(Sampler):
    def __init__(self,dataset,mode,seed,evidence):
        index=json.loads((ROOT/'data/yolo/attribute_index.json').read_text())['train']
        self.position={Path(p).stem:i for i,p in enumerate(dataset.im_files)}
        assert set(self.position)==set(index) and len(self.position)==len(dataset)
        self.index=index
        self.mode=mode
        self.seed=seed
        self.epoch=0
        self.evidence=Path(evidence)
        self.evidence.mkdir(exist_ok=True,parents=True)

    def set_epoch(self,epoch):
        self.epoch=epoch

    def __len__(self):
        return len(self.position)

    def __iter__(self):
        order=build_schedule(self.index,self.mode,self.seed,self.epoch)
        counts=Counter(order)
        assert max(counts.values())<=2
        negatives=[s for s in order if self.index[s]['exist']==0]
        assert len(negatives)==3612 and len(set(negatives))==3612
        if self.mode=='quota':
            assert Counter(self.index[s]['seq'] for s in order)==Counter(r['seq'] for r in self.index.values())
        text='\n'.join(order)+'\n'
        prefix=self.evidence/f'epoch_{self.epoch:03d}'
        prefix.with_suffix('.txt').write_text(text)
        record=dict(epoch=self.epoch,mode=self.mode,primary_slots=len(order),
            unique_primary_frames=len(counts),negative_primary_slots=len(negatives),
            tc_positive_slots=sum(self.index[s]['exist']==1 and self.index[s]['attr'].get('TC')==1 for s in order),
            sequence_counts=dict(Counter(self.index[s]['seq'] for s in order)),
            order_sha256=hashlib.sha256(text.encode()).hexdigest(),max_repeats=max(counts.values()),
            augmentation_note='Counts cover primary samples; standard Mosaic also draws secondary images from its buffer.')
        prefix.with_suffix('.json').write_text(json.dumps(record,indent=2))
        return iter([self.position[s] for s in order])


class EpochLoader(DataLoader):
    def reset(self):
        iterator=getattr(self,'_iterator',None)
        if iterator is not None:
            iterator._shutdown_workers()
            self._iterator=None

    def close(self):
        self.reset()


class InnovationTrainer(DetectionTrainer):
    def get_dataloader(self,dataset_path,batch_size=16,rank=-1,mode='train'):
        if mode!='train':
            return super().get_dataloader(dataset_path,batch_size,rank,mode)
        assert rank==-1, 'This isolated queue uses one GPU and two independent processes, not DDP'
        dataset=self.build_dataset(dataset_path,mode,batch_size)
        sampler=EpochSampler(dataset,ACTIVE_JOB['sampling'],self.args.seed,
                             STUDY/'jobs'/ACTIVE_JOB['run_name']/'sampling')
        generator=torch.Generator().manual_seed(6148914691236517205+self.args.seed)
        return EpochLoader(dataset,batch_size=batch_size,sampler=sampler,num_workers=self.args.workers,
            collate_fn=dataset.collate_fn,worker_init_fn=seed_worker,generator=generator,
            pin_memory=False,persistent_workers=True,prefetch_factor=1,multiprocessing_context='forkserver')


def main():
    global ACTIVE_JOB
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,default=STUDY)
    parser.add_argument('--name',required=True)
    parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args()
    contract=json.loads((args.study/'contract.json').read_text())
    job=next(j for j in contract['models'] if j['name']==args.name)
    run_name=job['name'] if args.attempt==1 else f"{job['name']}_attempt{args.attempt}"
    ACTIVE_JOB=dict(job,run_name=run_name)
    common=json.loads((args.study/'common_args.json').read_text())
    common.update(name=run_name,data=job['data'])
    assert not (Path(common['project'])/run_name).exists()
    assert hashlib.sha256(Path(job['weights']).read_bytes()).hexdigest()==job['sha256']
    assert hashlib.sha256((ROOT/'data/yolo/attribute_index.json').read_bytes()).hexdigest()==contract['index_sha256']
    for split in ('train','val'):
        p=ROOT/'experiments/baselines_100e_20261017'/f'{split}.txt'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==contract['splits'][split]['list_sha256']
    evidence=args.study/'jobs'/run_name
    evidence.mkdir(parents=True,exist_ok=False)
    manifest=dict(name=args.name,run_name=run_name,status='running',started_at=now(),pid=os.getpid(),
        config=common,job=job,contract_sha256=hashlib.sha256((args.study/'contract.json').read_bytes()).hexdigest(),
        environment=contract['environment'],argv=sys.argv,seed_role='exploratory_single_seed',
        temporal=job['temporal'],no_test_evaluation=True)
    mp=evidence/'manifest.json'
    mp.write_text(json.dumps(manifest,indent=2))
    if job['temporal']:
        import ultralytics.data.augment as augment
        augment.RandomHSV=SharedBrightness
        ready=json.loads((args.study/'temporal_data_ready.json').read_text())
        assert ready['status']=='PASS' and ready['splits']['train']['frames']==25612
    base_loader=build.InfiniteDataLoader
    class BoundedLoader(base_loader):
        def __init__(self,*pos,**kwargs):
            if kwargs.get('num_workers',0)>0:
                kwargs.update(prefetch_factor=1,multiprocessing_context='forkserver')
            kwargs['pin_memory']=False
            super().__init__(*pos,**kwargs)
    build.InfiniteDataLoader=BoundedLoader
    model=None
    try:
        model=YOLO(job['weights'])
        def ready(trainer):
            manifest.update(actual_parameters=sum(p.numel() for p in trainer.model.parameters()),
                actual_optimizer=type(trainer.optimizer).__name__,actual_train_frames=len(trainer.train_loader.dataset),
                actual_val_frames=len(trainer.test_loader.dataset))
            mp.write_text(json.dumps(manifest,indent=2))
        model.add_callback('on_train_start',ready)
        model.add_callback('on_train_epoch_start',lambda trainer:trainer.train_loader.sampler.set_epoch(trainer.epoch))
        def keep_ap50(trainer):
            score=float(trainer.metrics.get('metrics/mAP50(B)',-1))
            if score>manifest.get('peak_ap50',-1):
                shutil.copy2(trainer.last,trainer.wdir/'bestAP50.pt')
                manifest.update(peak_ap50=score,peak_ap50_epoch=int(trainer.epoch)+1)
                mp.write_text(json.dumps(manifest,indent=2))
        model.add_callback('on_model_save',keep_ap50)
        model.train(trainer=InnovationTrainer,**common)
        run=Path(common['project'])/run_name
        assert (run/'weights/best.pt').is_file() and (run/'results.csv').is_file()
        # Re-evaluate every run, not just a visually selected winner. No test access.
        sys.path.insert(0,str(ROOT/'experiments/baselines_100e_20261017'))
        import audit_validation as av
        av.INDEX=json.loads((ROOT/'data/yolo/attribute_index.json').read_text())['val']
        eval_model=YOLO(str(run/'weights/best.pt'))
        dest=Path(common['project'])/'validation_audit'/run_name
        metrics=eval_model.val(validator=av.AuditValidator,data=job['data'],split='val',imgsz=640,
            batch=128,workers=4,device='0',half=True,conf=0.001,iou=0.7,max_det=300,rect=True,
            plots=False,save_json=False,project=str(dest.parent),name=dest.name,exist_ok=False,verbose=False)
        audit=dict(name=run_name,weights_sha256=hashlib.sha256((run/'weights/best.pt').read_bytes()).hexdigest(),
            protocol=dict(split='val',frames=9066,includes_negatives=True,batch=128,half=True,
                rect=True,imgsz=640,conf=0.001,nms_iou=0.7,max_det=300,temporal=job['temporal'],
                checkpoint_selection='native best mAP50-95; AP50-maximum checkpoint is supplemental only'),
            metrics={k:float(v) for k,v in metrics.results_dict.items()},
            groups=json.loads((dest/'groups.json').read_text()))
        (dest/'audit.json').write_text(json.dumps(audit,indent=2))
        manifest.update(status='completed',ended_at=now(),last_epoch=int(model.trainer.epoch)+1,
                        output=str(run),completion_reason='training_and_uniform_val_audit_returned_successfully',
                        val_audit=str(dest/'audit.json'))
    except BaseException as exc:
        manifest.update(status='failed',ended_at=now(),error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        if model is not None and getattr(model,'trainer',None) is not None:
            for loader in [getattr(model.trainer,'train_loader',None),getattr(model.trainer,'test_loader',None)]:
                if isinstance(loader,EpochLoader):
                    loader.close()
                else:
                    iterator=getattr(loader,'iterator',None)
                    if iterator is not None and hasattr(iterator,'_shutdown_workers'):
                        iterator._shutdown_workers()
        mp.write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
