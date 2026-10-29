"""Real-image alignment, augmentation semantics, worker IPC and finite-gradient smoke checks."""
import copy
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import RandomHSV
from ultralytics.data.build import build_yolo_dataset,seed_worker
from ultralytics.nn.tasks import DetectionModel
from sampling import build_schedule
from train_innovation import EpochSampler,EpochLoader,SharedBrightness,ROOT,STUDY


def main():
    torch.set_num_threads(2)
    common=json.loads((STUDY/'common_args.json').read_text())
    contract=json.loads((STUDY/'contract.json').read_text())
    from ultralytics.data.utils import check_det_dataset
    # A replicated grayscale image must receive the same brightness mapping as the old HSV path.
    gray=np.tile(np.arange(256,dtype=np.uint8),(20,1))
    image=cv2.merge([gray,gray,gray])
    for seed in range(10):
        np.random.seed(seed); a=RandomHSV(.015,.7,.4)({'img':image.copy()})['img']
        np.random.seed(seed); b=SharedBrightness(.015,.7,.4)({'img':image.copy()})['img']
        assert np.array_equal(a,b),seed
    index=json.loads((ROOT/'data/yolo/attribute_index.json').read_text())['train']
    # Real-data checks cover all planned epoch draws, not just a miniature synthetic example.
    from collections import Counter
    original=Counter(r['seq'] for r in index.values())
    for epoch in range(100):
        a=build_schedule(index,'global',0,epoch); b=build_schedule(index,'quota',0,epoch)
        assert len(a)==len(b)==25612
        assert Counter(index[s]['seq'] for s in b)==original
        assert sum(index[s]['exist'] and index[s]['attr'].get('TC')==1 for s in a)==sum(index[s]['exist'] and index[s]['attr'].get('TC')==1 for s in b)
        for order in (a,b):
            assert max(Counter(order).values())<=2
            assert len({s for s in order if not index[s]['exist']})==3612
    records=[]
    source=YOLO(str(ROOT/'yolo11n.pt')).model
    for temporal in (False,True):
        cfg=get_cfg(overrides=common)
        cfg.workers=2
        cfg.batch=4
        cfg.data=str(STUDY/'cst_temporal.yaml') if temporal else common['data']
        data=check_det_dataset(cfg.data)
        if temporal:
            import ultralytics.data.augment as aug
            aug.RandomHSV=SharedBrightness
        dataset=build_yolo_dataset(cfg,data['train'],4,data,mode='train',rect=False,stride=32)
        sampler=EpochSampler(dataset,'quota',0,STUDY/'preflight_sampling'/str(temporal))
        loader=EpochLoader(dataset,batch_size=4,sampler=sampler,num_workers=2,collate_fn=dataset.collate_fn,
            worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(0),
            pin_memory=False,persistent_workers=True,prefetch_factor=1,multiprocessing_context='forkserver')
        batch=next(iter(loader))
        assert batch['img'].shape==(4,3,640,640)
        assert torch.isfinite(batch['bboxes']).all()
        loader.close()
        model=DetectionModel(copy.deepcopy(source.yaml),ch=3,nc=1,verbose=False)
        model.load(source,verbose=False)
        model=model.cuda().train()
        model.args=cfg
        for k,value in batch.items():
            if isinstance(value,torch.Tensor): batch[k]=value.cuda()
        batch['img']=batch['img'].float()/255
        with torch.autocast('cuda',dtype=torch.float16):
            loss,items=model(batch)
            scalar=loss.sum()/4
        scalar.backward()
        assert torch.isfinite(scalar)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        records.append(dict(temporal=temporal,batch_shape=list(batch['img'].shape),
            real_image_loss_debug_only=float(scalar.detach()),finite_gradient=True,
            parameters=sum(p.numel() for p in model.parameters()),worker_ipc=True))
        del model,batch
        torch.cuda.empty_cache()
    result=dict(status='PASS',real_image_smoke=records,all_100_primary_schedules_valid=True,
        shared_brightness_matches_grayscale_hsv=True,temporal_data_ready=json.loads((STUDY/'temporal_data_ready.json').read_text()))
    (STUDY/'preflight.json').write_text(json.dumps(result,indent=2))
    print('PREFLIGHT_PASS',json.dumps(result),flush=True)


if __name__=='__main__':main()
