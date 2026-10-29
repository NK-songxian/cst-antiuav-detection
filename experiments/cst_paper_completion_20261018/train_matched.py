"""Frozen input-only controls, with original full training schedule and native best.pt."""
import argparse,hashlib,json,os,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
import torch,cv2
from ultralytics import YOLO
from ultralytics.data import build
import ultralytics.data.augment as aug
import matched_runtime as rt
ROOT=rt.ROOT
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args()
    c=json.loads((a.study/'contract.json').read_text());job=next(j for j in c['models'] if j['name']==a.name)
    for path,h in c['code_sha256'].items():assert sha(path)==h,path
    assert sha(job['weights'])==job['weights_sha256']
    common=json.loads((a.study/'common_args.json').read_text())
    common.update(name=job['name'],seed=job['seed'],workers=job['workers'],data=job['data'])
    run=Path(common['project'])/job['name'];assert not run.exists(),run
    out=a.study/'jobs'/job['name'];out.mkdir(parents=True,exist_ok=False)
    mp=out/'manifest.json';m=dict(status='running',pid=os.getpid(),started_at=now(),job=job,config=common,
        contract_sha256=sha(a.study/'contract.json'),selection='native best.pt by val AP50-95',test_used_by_training=False,
        fairness='same PNG image list, same current labels, same sampler, same shared-brightness augmentation; input mode is the ablation')
    def save():
        tmp=mp.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2));tmp.replace(mp)
    save();torch.set_num_threads(2);cv2.setNumThreads(0)
    torch.cuda.set_per_process_memory_fraction(job['allocator_cap_bytes']/torch.cuda.get_device_properties(0).total_memory)
    rt.INPUT_MODE=job['input_mode'];rt.ti.STUDY=a.study
    rt.ti.ACTIVE_JOB=dict(run_name=job['name'],sampling='uniform')
    # Both arms use exactly the same transform, including the RNG draws.
    aug.RandomHSV=rt.ti.SharedBrightness
    original=build.InfiniteDataLoader
    class BoundedLoader(original):
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
            assert t.train_loader.dataset.input_mode==job['input_mode']
            m.update(actual_batch=t.batch_size,actual_optimizer=type(t.optimizer).__name__,
                actual_lr_groups=[g['lr'] for g in t.optimizer.param_groups],actual_train_workers=t.train_loader.num_workers,
                actual_val_workers=t.test_loader.num_workers,actual_val_batch=t.test_loader.batch_size,
                parameters=sum(p.numel() for p in t.model.parameters()),initial_accumulation=t.accumulate)
            save()
        def epoch_start(t):t.train_loader.sampler.set_epoch(t.epoch)
        def batch_end(t):assert t.batch_size==256,'Silent batch reduction invalidates the paired comparison'
        def epoch_end(t):
            m.update(latest_epoch=int(t.epoch)+1,peak_reserved_bytes=torch.cuda.max_memory_reserved(),updated_at=now());save()
        model.add_callback('on_train_start',ready);model.add_callback('on_train_epoch_start',epoch_start)
        model.add_callback('on_train_batch_end',batch_end);model.add_callback('on_train_epoch_end',epoch_end)
        model.train(trainer=rt.MatchedTrainer,**common)
        assert (run/'weights/best.pt').is_file()
        m.update(status='trained',ended_training_at=now(),last_epoch=int(model.trainer.epoch)+1,best_sha256=sha(run/'weights/best.pt'));save()
    except BaseException as e:
        m.update(status='failed',ended_at=now(),error=repr(e),traceback=traceback.format_exc());raise
    finally:
        if model is not None and getattr(model,'trainer',None) is not None:
            for loader in [getattr(model.trainer,'train_loader',None),getattr(model.trainer,'test_loader',None)]:
                if isinstance(loader,rt.ti.EpochLoader):loader.close()
                else:
                    it=getattr(loader,'iterator',None)
                    if it is not None and hasattr(it,'_shutdown_workers'):
                        try:it._shutdown_workers()
                        except Exception:pass
        save()
if __name__=='__main__':main()
