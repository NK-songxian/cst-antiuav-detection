"""Real batch256 resource probe before scheduling the corrected CST ablations."""
import argparse,copy,gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import init_seeds

ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
AAT=ROOT/'experiments/aat_temporal_100e_20261017'
sys.path.insert(0,str(AAT))
import train_innovation as ti


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);init_seeds(0,deterministic=True)
    import ultralytics.data.augment as aug
    aug.RandomHSV=ti.SharedBrightness
    common=json.loads((AAT/'common_args.json').read_text())
    common.update(data=str(AAT/'cst_temporal.yaml'),workers=0)
    cfg=get_cfg(overrides=common);data=check_det_dataset(cfg.data)
    ds=build_yolo_dataset(cfg,data['train'],256,data,mode='train',rect=False,stride=32)
    batch=ds.collate_fn([ds[i] for i in range(256)])
    batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    batch['img']=batch['img'].float()/255
    results=[]
    for key,filename,cap_gb in [('yolov8n','yolov8n.pt',35),('yolo11n','yolo11n.pt',38),('yolo11s','yolo11s.pt',68)]:
        torch.cuda.empty_cache();torch.cuda.set_per_process_memory_fraction(cap_gb*1024**3/torch.cuda.get_device_properties(0).total_memory)
        r=dict(model=key,batch=256,imgsz=640,allocator_cap_gib=cap_gb,kind='resource_not_accuracy')
        try:
            source=YOLO(str(ROOT/filename)).model
            model=DetectionModel(copy.deepcopy(source.yaml),nc=1,verbose=False)
            model.load(source,verbose=False);model=model.cuda().train();model.args=cfg
            optimizer=torch.optim.SGD(model.parameters(),lr=.001,momentum=.937)
            scaler=torch.amp.GradScaler('cuda');times=[];torch.cuda.reset_peak_memory_stats()
            for step in range(2):
                optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.float16):loss,items=model(batch);scalar=loss.sum()
                assert torch.isfinite(scalar);scaler.scale(scalar).backward();scaler.step(optimizer);scaler.update()
                torch.cuda.synchronize();times.append(time.perf_counter()-t)
            r.update(status='PASS',peak_reserved=torch.cuda.max_memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),step_seconds=times,
                     parameters=sum(p.numel() for p in model.parameters()))
        except BaseException as e:
            r.update(status='FAIL',error=repr(e));results.append(r)
            (a.study/'resource_probes.json').write_text(json.dumps(results,indent=2));raise
        results.append(r);print(json.dumps(r),flush=True)
        (a.study/'resource_probes.json').write_text(json.dumps(results,indent=2))
        del model,source,optimizer,scaler,loss,scalar;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
