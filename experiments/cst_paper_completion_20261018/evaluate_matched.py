"""All backgrounds retained, fixed val-AP5095 checkpoint, no test model selection."""
import argparse,json,hashlib,importlib.util,sys
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
    c=json.loads((a.study/'contract.json').read_text());job=next(j for j in c['models'] if j['name']==a.name)
    common=json.loads((a.study/'common_args.json').read_text());root=Path(common['project'])
    weights=Path(job.get('trained_weights',str(root/job['name']/'weights/best.pt')));assert weights.is_file()
    prior.INDEX=json.loads((ROOT/'data/yolo/attribute_index.json').read_text())[a.split];prior.av.INDEX=prior.INDEX
    prior.REPEAT_CURRENT=job['input_mode']=='current'
    folder='validation_audit' if a.split=='val' else 'test_audit';dest=root/folder/job['name'];assert not dest.exists(),dest
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(10*1024**3/torch.cuda.get_device_properties(0).total_memory)
    model=YOLO(str(weights));params=sum(p.numel() for p in model.model.parameters())
    m=model.val(validator=prior.CompletedValidator,data=str(OLD/'cst_temporal_evaluation.yaml'),split=a.split,
        imgsz=640,batch=128,workers=0,device='0',half=True,conf=.001,iou=.7,max_det=300,rect=True,
        plots=False,save_json=False,project=str(dest.parent),name=dest.name,exist_ok=False,verbose=False)
    report=dict(name=job['name'],input_mode=job['input_mode'],trained_current_only_control=prior.REPEAT_CURRENT,
        seed=job['seed'],weights_sha256=sha(weights),parameters_before_fusion=params,
        script_sha256=sha(__file__),validator_source_sha256=sha(ROOT/'experiments/baselines_100e_20261017/audit_validation.py'),
        completed_at=datetime.now(timezone.utc).isoformat(),protocol=dict(split=a.split,frames=len(prior.INDEX),includes_negatives=True,
          imgsz=640,batch=128,workers=0,half=True,rect=True,conf=.001,nms_iou=.7,max_det=300,
          actual_shapes=m.audit_shapes,selection='native best.pt selected by val AP50-95; no test selection'),
        metrics={k:float(v) for k,v in m.results_dict.items()},groups=json.loads((dest/'groups.json').read_text()))
    (dest/'audit.json').write_text(json.dumps(report,indent=2));print('AUDIT_COMPLETE',job['name'],a.split,report['metrics'],flush=True)
if __name__=='__main__':main()
