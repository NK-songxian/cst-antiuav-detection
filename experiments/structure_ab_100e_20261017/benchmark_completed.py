"""Isolated, batch-one PyTorch benchmark; not an edge-device deployment claim."""
import hashlib
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import get_flops

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
RUNS = ROOT/'runs/structure_ab_100e_20261017'
OUT = ROOT/'experiments/structure_ab_100e_20261017'
cv2.setNumThreads(1)
torch.set_num_threads(2)

def describe(x):
    x = np.asarray(x)
    return dict(n=len(x),mean_ms=float(x.mean()),median_ms=float(np.median(x)),
                p95_ms=float(np.quantile(x,.95)))

def main():
    LOGGER.setLevel(logging.ERROR)
    weights = {'yolo11n':ROOT/'runs/baselines_100e_20261017/yolo11n/weights/best.pt',
               'A_zero_deploy':RUNS/'A_zero_deploy/weights/deploy.pt',
               'B_p2_compact':RUNS/'B_p2_compact/weights/deploy.pt'}
    pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    assert not pids, 'GPU busy before benchmark: '+pids
    paths = (ROOT/'experiments/baselines_100e_20261017/val.txt').read_text().splitlines()
    sample = np.linspace(0,len(paths)-1,32,dtype=int)
    images = [cv2.imread(paths[i]) for i in sample]
    assert all(im is not None for im in images)
    inputs=[]
    for im in images:
        h,w=im.shape[:2]
        assert (h,w)==(512,640),(h,w)
        im=cv2.copyMakeBorder(im,64,64,0,0,cv2.BORDER_CONSTANT,value=(114,114,114))
        inputs.append(torch.from_numpy(np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1))).unsqueeze(0).cuda().half()/255)
    models={}
    rows={}
    for name,path in weights.items():
        yolo=YOLO(str(path))
        yolo.model.fuse(verbose=False)
        rows[name]={'weights':str(path),'weights_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'fused_parameters':sum(p.numel() for p in yolo.model.parameters()),
                    'gflops_640':get_flops(yolo.model,imgsz=640)}
        yolo.model.cuda().half().eval()
        models[name]=yolo
    with torch.inference_mode():
        for model in models.values():
            for i in range(40):model.model(inputs[i%32])
        torch.cuda.synchronize()
        raw={name:[] for name in models}
        # Round robin makes thermal/clock drift affect all models similarly.
        for i in range(200):
            for name,model in models.items():
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record(); result=model.model(inputs[i%32]);end.record()
                raw[name].append((start,end))
        torch.cuda.synchronize()
        for name,pairs in raw.items():rows[name]['raw_cuda_forward']=describe([a.elapsed_time(b) for a,b in pairs])
        predict_args=dict(device='0',half=True,imgsz=640,rect=False,conf=.25,iou=.7,max_det=300,
                          verbose=False,save=False,stream=False)
        for model in models.values():
            for i in range(12):model.predict(images[i%32],**predict_args)
        timing={name:[] for name in models}
        for i in range(128):
            for name,model in models.items():
                torch.cuda.synchronize();start=time.perf_counter()
                result=model.predict(images[i%32],**predict_args)
                torch.cuda.synchronize();timing[name].append((time.perf_counter()-start)*1000)
        for name,x in timing.items():rows[name]['cpu_array_to_boxes_wall']=describe(x)
    report={'recorded_at':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
            'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
            'protocol':{'batch':1,'network_shape':[1,3,640,640],'precision':'FP16',
                'backend':'PyTorch eager, fused, no TensorRT, no CUDA graphs',
                'raw':'GPU-resident real input tensor to raw predictions; CUDA events',
                'wall':'decoded CPU BGR array through YOLO.predict preprocessing/H2D/network/NMS; synchronized wall time; JPEG decoding excluded',
                'images':32,'raw_warmup_per_model':40,'raw_samples_per_model':200,
                'wall_warmup_per_model':12,'wall_samples_per_model':128,
                'conf':.25,'iou':.7,'max_det':300,'schedule':'round-robin; one process; no concurrent evaluation',
                'logging':'Ultralytics warnings suppressed consistently for all models',
                'sample_paths':[paths[i] for i in sample]},'models':rows}
    (OUT/'completed_benchmark.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
