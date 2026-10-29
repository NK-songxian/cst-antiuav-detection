"""Check original best checkpoints against exported deployment on real images."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY=ROOT/'experiments/structure_ab_100e_20261017'
sys.path.insert(0,str(STUDY))
cv2.setNumThreads(1)
torch.set_num_threads(2)
precision_before={'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                  'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
paths=(ROOT/'experiments/baselines_100e_20261017/val.txt').read_text().splitlines()
selected=[paths[i] for i in np.linspace(0,len(paths)-1,16,dtype=int)]
rows={}
with torch.inference_mode():
    for name in ['A_zero_deploy','B_p2_compact']:
        folder=ROOT/'runs/structure_ab_100e_20261017'/name/'weights'
        ckpt=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        original=(ckpt.get('ema') or ckpt['model']).float().eval().cuda()
        ckpt=torch.load(folder/'deploy.pt',map_location='cpu',weights_only=False)
        deployed=(ckpt.get('ema') or ckpt['model']).float().eval().cuda()
        errors=[]
        for filename in selected:
            im=cv2.imread(filename)
            im=cv2.copyMakeBorder(im,64,64,0,0,cv2.BORDER_CONSTANT,value=(114,114,114))
            x=torch.from_numpy(np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1))).unsqueeze(0).cuda().float()/255
            a,b=original(x)[0],deployed(x)[0]
            torch.testing.assert_close(a,b,atol=1e-2,rtol=1e-4)
            errors.append({'coordinate_max_abs':float((a[:,:4]-b[:,:4]).abs().max()),
                           'class_conf_max_abs':float((a[:,4:]-b[:,4:]).abs().max())})
        rows[name]={'images':16,'fp32_allclose_pass':True,'atol':1e-2,'rtol':1e-4,
                    'coordinate_max_abs':max(e['coordinate_max_abs'] for e in errors),
                    'class_conf_max_abs':max(e['class_conf_max_abs'] for e in errors)}
        del original,deployed,a,b,x,ckpt
        torch.cuda.empty_cache()
result={'protocol':'16 deterministic real validation images; 640x640 padded input; strict CUDA FP32 with TF32 disabled; best before folding vs exported deploy; raw output tolerance, not exact FP16 or NMS identity guarantee',
        'precision_before':precision_before,
        'initial_check_failure':{'protocol':'CUDA float tensors with backend defaults; TF32 state not recorded during initial failed run',
             'mismatched_elements':5473,'total_elements':42000,'max_abs_error':0.9854736328125,
             'note':'Initial assertion failed; strict FP32 rerun is required. This record does not by itself establish a TF32 root cause.'},
        'paths':selected,'models':rows}
(STUDY/'completed_folding_equivalence.json').write_text(json.dumps(result,indent=2))
print(json.dumps(rows,indent=2))
