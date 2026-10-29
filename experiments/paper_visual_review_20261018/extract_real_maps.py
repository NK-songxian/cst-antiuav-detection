"""CPU-only checkpoint diagnostics; no training changes and no synthetic heatmaps."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
import cv2,torch
from ultralytics import YOLO
ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
OLD=ROOT/'experiments/aat_temporal_100e_20261017/completed_review_20261017'
OUT=ROOT/'experiments/paper_visual_review_20261018';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    cases=json.loads((OLD/'qualitative/cases.json').read_text())['cases']
    temporal={Path(p).stem:Path(p) for p in (OLD/'temporal_test.txt').read_text().splitlines()}
    single={Path(p).stem:Path(p) for p in (ROOT/'experiments/baselines_100e_20261017/test.txt').read_text().splitlines()}
    weights={'single':ROOT/'runs/baselines_100e_20261017/yolo11n/weights/best.pt','temporal':ROOT/'runs/aat_temporal_100e_20261017/C1_temporal_uniform/weights/best.pt'}
    records=[];arrays={}
    for kind,wp in weights.items():
        model=YOLO(str(wp)).model.cpu().float().eval()
        for p in model.parameters():p.requires_grad_(True)
        head=model.model[-1];saved={}
        def pre(m,args):saved['feature']=args[0][0];saved['feature'].retain_grad()
        def cls(m,args,output):saved['class_logits']=output
        hooks=[head.register_forward_pre_hook(pre),head.cv3[0].register_forward_hook(cls)]
        for ci,case in enumerate(cases):
            stem=case['selected'];path=(single if kind=='single' else temporal)[stem];bgr=cv2.imread(str(path));assert bgr is not None
            H,W=bgr.shape[:2]
            if case['yolo_ground_truth'].strip():
                _,x,y,w,h=map(float,case['yolo_ground_truth'].splitlines()[0].split());target=(x*W,y*H);rule='current GT centre; diagnostic target selection only'
            else:
                boxes=[]
                for name in ['yolo11n','C1_temporal_uniform']:
                    p=case['models'][name]['predictions'];boxes.extend(zip(p['conf'],p['xyxy']))
                _,b=max(boxes,key=lambda p:p[0]);target=((b[0]+b[2])/2,(b[1]+b[3])/2);rule='highest-scored FP box centre from either model; common target for both'
            ratio=min(544/H,672/W,1.0);nw,nh=round(W*ratio),round(H*ratio);left=(672-nw)//2;top=(544-nh)//2
            image=cv2.copyMakeBorder(cv2.resize(bgr,(nw,nh)),top,544-nh-top,left,672-nw-left,cv2.BORDER_CONSTANT,value=(114,114,114))
            tensor=torch.from_numpy(image[:,:,::-1].transpose(2,0,1).copy()).float()[None]/255
            for mode in ([kind,'repeat_current'] if kind=='temporal' else [kind]):
                x=tensor.clone()
                if mode=='repeat_current':x=x[:,0:1].repeat(1,3,1,1)
                model.zero_grad(set_to_none=True);saved.clear();model(x)
                logits=saved['class_logits'];feat=saved['feature'];fh,fw=logits.shape[-2:]
                gx=min(fw-1,max(0,int((target[0]*ratio+left)/672*fw)));gy=min(fh-1,max(0,int((target[1]*ratio+top)/544*fh)))
                score=logits[0,0,gy,gx];score.backward()
                grad=feat.grad;assert grad is not None and torch.isfinite(grad).all() and grad.abs().max()>0
                cam=torch.relu((grad.mean((2,3),keepdim=True)*feat).sum(1))[0].detach().numpy()
                prob=logits.detach().sigmoid()[0,0].numpy()
                def restore(a):
                    big=cv2.resize(a,(672,544),interpolation=cv2.INTER_LINEAR)
                    return cv2.resize(big[top:top+nh,left:left+nw],(W,H),interpolation=cv2.INTER_LINEAR)
                key=f'case{ci}_{mode}';arrays[key+'_cam']=restore(cam);arrays[key+'_prob']=restore(prob)
                records.append(dict(key=key,case=stem,kind=mode,weights_sha256=sha(wp),image_sha256=sha(path),input_tensor_sha256=hashlib.sha256(x.detach().numpy().tobytes()).hexdigest(),input_shape=list(x.shape),feature_shape=list(feat.shape),target_xy=target,target_cell=[gy,gx],target_rule=rule,target_logit=float(score.detach()),target_probability=float(score.detach().sigmoid()),gradient_l1=float(grad.abs().sum()),cam_min=float(cam.min()),cam_max=float(cam.max()),layer='Detect input P3 (stride8); class branch cv3[0]',map='ReLU(sum_channels(mean_spatial(d logit / d feature) * feature))',diagnostic='FP32 CPU; fixed shape544x672; probe score is not the NMS box confidence'))
                print('MAP_COMPLETE',key,records[-1]['target_probability'],flush=True)
        for h in hooks:h.remove()
        del model
    np.savez_compressed(OUT/'real_maps.npz',**arrays)
    (OUT/'map_provenance.json').write_text(json.dumps({'records':records,'cam_warning':'Spatial attribution for a fixed P3 cell; not a temperature map, causal proof, or comparable cross-model amplitude. Repeat-current is an OOD intervention.','npz_sha256':sha(OUT/'real_maps.npz'),'script_sha256':sha(__file__)},indent=2))
    with zipfile.ZipFile(OUT/'real_maps.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['real_maps.npz','map_provenance.json','extract_real_maps.py']:z.write(OUT/name,name)
    print('REAL_MAPS_READY',flush=True)
if __name__=='__main__':main()
