"""Read frozen test data/weights and export two reappearance examples, no training."""
from pathlib import Path
import json,hashlib,zipfile
import cv2,numpy as np,torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

R=Path('/root/autodl-tmp/Ryan/cst_antiuav')
OUT=R/'paper_assets_20261018_v7';OUT.mkdir(exist_ok=True)
CASES=json.loads((OUT/'reappearance_candidates_v7.json').read_text())
TEST_LIST=R/'experiments/aat_temporal_100e_20261017/completed_review_20261017/temporal_test.txt'
PATHS={Path(s).stem:Path(s) for s in TEST_LIST.read_text().splitlines()}
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(min(.15,2*1024**3/torch.cuda.get_device_properties(0).total_memory))
validator=DetectionValidator(args={'conf':.001,'iou':.7,'max_det':300})
validator.iouv=validator.iouv.to('cuda')
weights={m:R/f'runs/cst_paper_completion_20261018/{n}/weights/best.pt' for m,n in [('single','N_current_seed0'),('temporal','N_tcs_seed0')]}
models={m:YOLO(str(p)) for m,p in weights.items()}
stems=set(['building_66_f000010','building_66_f000015','building_66_f000020'])
for case in CASES.values():
    e=case['event'];stems.update(e['stems'])
    stems.update(f"{e['seq']}_f{e['start_frame']-gap:06d}" for gap in [10,5])
frames={}
for stem in sorted(stems):
    p=PATHS[stem];im=cv2.imread(str(p));assert im is not None,p
    h,w=im.shape[:2];assert (h,w)==(512,640)
    cv2.imwrite(str(OUT/f'{stem}.png'),im[:,:,2])
    lp=Path(str(p).replace('/images/','/labels/')).with_suffix('.txt');text=lp.read_text();gt=[]
    for line in text.splitlines():
        _,x,y,bw,bh=map(float,line.split());gt.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
    f={'gt_xyxy':gt,'shape':[h,w],'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'predictions':{}}
    if stem.startswith('building_66_'):frames[stem]=f;continue
    for mode,model in models.items():
        rgb=im[:,:,::-1].copy()
        if mode=='single':rgb[:,:,:]=rgb[:,:,:1]
        padded=cv2.copyMakeBorder(rgb,16,16,16,16,cv2.BORDER_CONSTANT,value=(114,114,114))
        tensor=torch.from_numpy(np.ascontiguousarray(padded.transpose(2,0,1))).to('cuda').half().div_(255).unsqueeze(0)
        result=model.predict(tensor,conf=.001,iou=.7,max_det=300,half=True,verbose=False)[0]
        b=result.boxes;gt_tensor=torch.tensor(gt,dtype=torch.float32,device='cuda').reshape(-1,4)+16
        tp=validator._process_batch({'bboxes':b.xyxy,'cls':b.cls},{'bboxes':gt_tensor,'cls':torch.zeros(len(gt),device='cuda')})['tp']
        if torch.is_tensor(tp):tp=tp.cpu().numpy()
        tau=CASES['immediate']['thresholds']['单帧seed0' if mode=='single' else '时序seed0']
        conf=b.conf.cpu().numpy();selected=conf>=tau
        f['predictions'][mode]={'threshold':tau,'hit':bool(np.any(tp[:,0]&selected)),'boxes_xyxy':(b.xyxy.cpu().numpy()[selected]-16).tolist(),'conf':conf[selected].tolist()}
    frames[stem]=f
checks=[]
for key,c in CASES.items():
    for mode in ['single','temporal']:
        actual=[frames[s]['predictions'][mode]['hit'] for s in c['event']['stems']]
        checks.append({'case':key,'mode':mode,'expected':c[mode]['hits'],'actual':actual,'matches':actual==c[mode]['hits']})
record={'cases':CASES,'frames':frames,'checks':checks,'weights':{m:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for m,p in weights.items()},'protocol':'Frozen seed0 weights; fixed544x672 validation geometry (16pixel border around640x512 PNG), FP16;conf0.001;NMS0.7;maxDet300;display thresholds calibrated on validation5% budget. No model fitting.'}
(OUT/'assets.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
with zipfile.ZipFile(OUT/'assets.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.glob('*.png'):z.write(p,p.name)
    z.write(OUT/'assets.json','assets.json')
print('VISUAL_ASSETS',len(frames),'FRAMES',checks,flush=True)
