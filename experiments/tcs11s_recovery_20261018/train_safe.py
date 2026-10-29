"""C1 uniform sampler and temporal augmentation; bounded validation and forkserver loaders."""
import argparse,hashlib,json,os,shutil,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
import torch
from ultralytics import YOLO
from ultralytics.data import build

ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
AAT=ROOT/'experiments/aat_temporal_100e_20261017'
sys.path.insert(0,str(AAT))
import train_innovation as ti


def now():return datetime.now(timezone.utc).isoformat()


class SafeTrainer(ti.InnovationTrainer):
    def get_dataloader(self,dataset_path,batch_size=16,rank=-1,mode='train'):
        if mode!='train':batch_size=min(batch_size,128)
        return super().get_dataloader(dataset_path,batch_size,rank,mode)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    c=json.loads((a.study/'contract.json').read_text());job=next(j for j in c['models'] if j['name']==a.name)
    for f,h in c['code_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    assert hashlib.sha256(Path(job['weights']).read_bytes()).hexdigest()==job['sha256']
    common=json.loads((a.study/'common_args.json').read_text());common.update(name=a.name,data=job['data'])
    run=Path(common['project'])/a.name;assert not run.exists(),run
    evidence=a.study/'jobs'/a.name;evidence.mkdir(parents=True,exist_ok=False);mp=evidence/'manifest.json'
    m=dict(status='running',name=a.name,pid=os.getpid(),started_at=now(),config=common,job=job,
        contract_sha256=hashlib.sha256((a.study/'contract.json').read_bytes()).hexdigest(),test_used=False)
    def save():mp.write_text(json.dumps(m,indent=2))
    save();torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(job['allocator_cap_bytes']/torch.cuda.get_device_properties(0).total_memory)
    ti.STUDY=a.study;ti.ACTIVE_JOB=dict(job,run_name=a.name,sampling='uniform',temporal=True)
    import ultralytics.data.augment as aug
    if job['temporal']:aug.RandomHSV=ti.SharedBrightness
    parent=build.InfiniteDataLoader
    class BoundedLoader(parent):
        def __init__(self,*pos,**kw):
            if kw.get('num_workers',0)>0:kw.update(prefetch_factor=1,multiprocessing_context='forkserver')
            kw['pin_memory']=False
            super().__init__(*pos,**kw)
    build.InfiniteDataLoader=BoundedLoader
    model=None
    try:
        model=YOLO(job['weights'])
        def ready(t):
            assert t.batch_size==256 and t.args.imgsz==640
            assert len(t.train_loader.dataset)==25612 and len(t.test_loader.dataset)==9066
            m.update(actual_batch=t.batch_size,actual_optimizer=type(t.optimizer).__name__,
                parameters=sum(p.numel() for p in t.model.parameters()),train_workers=t.train_loader.num_workers,
                val_batch=t.test_loader.batch_size,train_frames=len(t.train_loader.dataset),val_frames=len(t.test_loader.dataset))
            save()
        def epoch_start(t):t.train_loader.sampler.set_epoch(t.epoch)
        def epoch_end(t):
            assert t.batch_size==256,'Unexpected batch auto-reduction'
            m.update(latest_epoch=int(t.epoch)+1,peak_reserved=torch.cuda.max_memory_reserved());save()
        def keep_ap50(t):
            score=float(t.metrics.get('metrics/mAP50(B)',-1))
            if score>m.get('peak_ap50',-1):
                shutil.copy2(t.last,t.wdir/'bestAP50.pt');m.update(peak_ap50=score,peak_ap50_epoch=int(t.epoch)+1);save()
        model.add_callback('on_train_start',ready);model.add_callback('on_train_epoch_start',epoch_start)
        model.add_callback('on_train_epoch_end',epoch_end);model.add_callback('on_model_save',keep_ap50)
        model.train(trainer=SafeTrainer,**common)
        assert (run/'weights/best.pt').is_file() and (run/'results.csv').is_file()
        sys.path.insert(0,str(ROOT/'experiments/baselines_100e_20261017'))
        import audit_validation as av
        av.INDEX=json.loads((ROOT/'data/yolo/attribute_index.json').read_text())['val']
        dest=Path(common['project'])/'validation_audit'/a.name
        metrics=YOLO(str(run/'weights/best.pt')).val(validator=av.AuditValidator,data=job['data'],split='val',
            imgsz=640,batch=128,workers=2,device='0',half=True,conf=.001,iou=.7,max_det=300,rect=True,
            plots=False,save_json=False,project=str(dest.parent),name=dest.name,exist_ok=False,verbose=False)
        audit=dict(name=a.name,weights_sha256=hashlib.sha256((run/'weights/best.pt').read_bytes()).hexdigest(),
            metrics={k:float(v) for k,v in metrics.results_dict.items()},groups=json.loads((dest/'groups.json').read_text()),
            protocol=dict(split='val',frames=9066,imgsz=640,batch=128,rect=True,half=True,conf=.001,iou=.7,max_det=300,
                checkpoint='native best.pt by val AP50-95',input_channels=job['temporal_channels']))
        (dest/'audit.json').write_text(json.dumps(audit,indent=2))
        m.update(status='completed',ended_at=now(),last_epoch=int(model.trainer.epoch)+1,output=str(run),val_audit=str(dest/'audit.json'))
    except BaseException as e:
        m.update(status='failed',ended_at=now(),error=repr(e),traceback=traceback.format_exc());raise
    finally:
        if model is not None and getattr(model,'trainer',None) is not None:
            for loader in [getattr(model.trainer,'train_loader',None),getattr(model.trainer,'test_loader',None)]:
                if isinstance(loader,ti.EpochLoader):loader.close()
                else:
                    it=getattr(loader,'iterator',None)
                    if it is not None and hasattr(it,'_shutdown_workers'):
                        try:it._shutdown_workers()
                        except Exception:pass
        save()


if __name__=='__main__':main()
