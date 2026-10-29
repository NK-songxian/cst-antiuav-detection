"""Same frozen matcher/evaluation settings, new causal input offset only."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
from datetime import datetime,timezone
import torch
from ultralytics import YOLO
ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
OLD=ROOT/'experiments/aat_temporal_100e_20261017/completed_review_20261017'
spec=importlib.util.spec_from_file_location('prior_eval',OLD/'evaluate_cst_completed.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--split',choices=['val','test'],required=True);a=p.parse_args()
    c=json.loads((a.study/'contract.json').read_text());job=next(x for x in c['models'] if x['name']==a.name)
    for path,h in c['code_sha256'].items():assert sha(path)==h,path
    common=json.loads((a.study/'common_args.json').read_text());root=Path(common['project']);weights=root/a.name/'weights/best.pt'
    prior.INDEX=json.loads((ROOT/'data/yolo/attribute_index.json').read_text())[a.split];prior.av.INDEX=prior.INDEX;prior.REPEAT_CURRENT=False
    dest=root/('validation_audit' if a.split=='val' else 'test_audit')/a.name
    assert not dest.exists();assert weights.is_file()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(10*1024**3/torch.cuda.get_device_properties(0).total_memory)
    model=YOLO(str(weights))
    metrics=model.val(validator=prior.CompletedValidator,data=job['data'],split=a.split,imgsz=640,batch=128,workers=0,device='0',half=True,conf=.001,iou=.7,max_det=300,rect=True,plots=False,save_json=False,project=str(dest.parent),name=dest.name,exist_ok=False,verbose=False)
    assert metrics.audit_shapes=={'(3, 544, 672)':len(prior.INDEX)},metrics.audit_shapes
    report=dict(name=a.name,split=a.split,far=job['far'],seed=job['seed'],weights_sha256=sha(weights),script_sha256=sha(__file__),completed_at=datetime.now(timezone.utc).isoformat(),protocol=dict(batch=128,imgsz=640,actual_shapes=metrics.audit_shapes,half=True,conf=.001,nms_iou=.7,max_det=300,rect=True,selection='native val AP50-95 best.pt, no test selection'),metrics={k:float(v) for k,v in metrics.results_dict.items()},groups=json.loads((dest/'groups.json').read_text()))
    (dest/'audit.json').write_text(json.dumps(report,indent=2))
    print('AUDIT_COMPLETE',a.name,a.split,report['metrics'],flush=True)
if __name__=='__main__':main()
