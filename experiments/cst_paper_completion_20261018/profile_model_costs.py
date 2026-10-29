"""CPU operation profiling of actual completed checkpoints at a declared square input."""
from pathlib import Path
import json,hashlib,torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops
ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
def main():
    torch.set_num_threads(2);rows={};shapes={}
    names=[(n,'baselines_100e_20261017') for n in ['yolov5nu','yolov8n','yolo26n','yolo11n']]
    names += [('C1_temporal_uniform','aat_temporal_100e_20261017'),('tcs_yolo8n','tcs_ablation_relaunch_20261017')]
    for n,study in names:
        p=ROOT/'runs'/study/n/'weights/best.pt';m=YOLO(str(p)).model.eval().cpu()
        row=dict(weights_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),parameters_before_fusion=sum(x.numel() for x in m.parameters()))
        try:
            v=float(get_flops(m,imgsz=640));row.update(gflops_3x640x640=v if v>0 else None,status='MEASURED' if v>0 else 'UNAVAILABLE')
        except Exception as e:row.update(gflops_3x640x640=None,status='UNAVAILABLE',error=repr(e))
        shapes[n]={k:list(v.shape) for k,v in m.state_dict().items()};rows[n]=row;print('COST',n,row,flush=True)
    report=dict(protocol='Ultralytics get_flops on actual unfused CPU checkpoint, 3x640x640 input; operation count only, not measured deployment latency or energy.',
        models=rows,same_state_shapes=dict(v8=shapes['yolov8n']==shapes['tcs_yolo8n'],v11=shapes['yolo11n']==shapes['C1_temporal_uniform']))
    (Path(__file__).parent/'model_costs.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
