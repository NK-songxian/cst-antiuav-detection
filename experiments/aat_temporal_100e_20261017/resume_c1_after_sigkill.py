"""Recover C1 from the intact epoch-42 checkpoint; preserve protocol and record reduced loader concurrency."""
import hashlib
import json
import os
import shutil
import sys
import traceback
from datetime import datetime,timezone
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.data import build
import ultralytics.data.augment as augment

ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY=ROOT/'experiments/aat_temporal_100e_20261017'
RUN=ROOT/'runs/aat_temporal_100e_20261017/C1_temporal_uniform'
DEST=STUDY/'recovery_C1_20261017'
sys.path.insert(0,str(STUDY))
import train_innovation as original


def now():return datetime.now(timezone.utc).isoformat()


def main():
    DEST.mkdir(exist_ok=False)
    gpu_free=torch.cuda.mem_get_info()[0]
    limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
    current=int(Path('/sys/fs/cgroup/memory.current').read_text())
    assert gpu_free>35*1024**3, 'Not enough GPU memory to resume the original batch256'
    assert limit-current>12*1024**3, 'Not enough container RAM headroom for a low-worker resume'
    last=RUN/'weights/last.pt'
    for src in [last,RUN/'weights/best.pt',RUN/'weights/bestAP50.pt',RUN/'results.csv',RUN/'args.yaml']:
        if src.exists():shutil.copy2(src,DEST/src.name)
    ck=torch.load(last,map_location='cpu',weights_only=False)
    assert ck['epoch']==41 and ck['optimizer'] is not None and ck.get('ema') is not None
    assert ck['train_args']['batch']==256 and ck['train_args']['imgsz']==640
    manifest=dict(status='running',pid=os.getpid(),started_at=now(),source_checkpoint=str(last),
        source_sha256=hashlib.sha256((DEST/'last.pt').read_bytes()).hexdigest(),checkpoint_epoch=42,
        expected_resume_epoch=43,original_exit_code=-9,original_exit_at='2026-10-17T05:52:28.054939+00:00',
        cause='SIGKILL sender unknown; no visible cgroup OOM kill event; kernel log unavailable',
        gpu_free_before_bytes=gpu_free,container_memory_before_bytes=current,container_memory_limit_bytes=limit,
        changes={'train_workers':[16,2],'val_workers':[32,4]},
        unchanged=['batch256','imgsz640','dataset','uniform sampler','temporal inputs','loss','optimizer checkpoint','EMA checkpoint','epoch limit100','patience20'],
        reproducibility='Resume preserves optimizer and EMA; changing workers and restarting RNG does not promise bitwise continuation')
    mp=DEST/'manifest.json';mp.write_text(json.dumps(manifest,indent=2))
    original.ACTIVE_JOB=dict(name='C1_temporal_uniform',run_name='C1_recovery_from_epoch43',sampling='uniform',temporal=True)
    augment.RandomHSV=original.SharedBrightness
    base=build.InfiniteDataLoader
    class BoundedLoader(base):
        def __init__(self,*pos,**kwargs):
            if kwargs.get('num_workers',0)>0:kwargs.update(prefetch_factor=1,multiprocessing_context='forkserver')
            kwargs['pin_memory']=False
            super().__init__(*pos,**kwargs)
    build.InfiniteDataLoader=BoundedLoader
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(35*1024**3/torch.cuda.get_device_properties(0).total_memory)
    del ck
    model=YOLO(str(last))
    def ready(trainer):
        assert trainer.start_epoch==42
        assert trainer.batch_size==256 and trainer.args.imgsz==640
        manifest.update(actual_start_epoch=trainer.start_epoch+1,actual_train_workers=trainer.train_loader.num_workers,
            actual_val_workers=trainer.test_loader.num_workers,actual_save_dir=str(trainer.save_dir))
        assert Path(trainer.save_dir).resolve()==RUN.resolve(),trainer.save_dir
        mp.write_text(json.dumps(manifest,indent=2))
    def epoch_done(trainer):
        manifest.update(latest_epoch=int(trainer.epoch)+1,updated_at=now())
        mp.write_text(json.dumps(manifest,indent=2))
    model.add_callback('on_train_start',ready)
    model.add_callback('on_train_epoch_start',lambda trainer:trainer.train_loader.sampler.set_epoch(trainer.epoch))
    model.add_callback('on_train_epoch_end',epoch_done)
    try:
        model.train(trainer=original.InnovationTrainer,resume=True,workers=2,device='0')
        manifest.update(status='completed',ended_at=now(),last_epoch=int(model.trainer.epoch)+1)
    except BaseException as exc:
        manifest.update(status='failed',ended_at=now(),error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        mp.write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
