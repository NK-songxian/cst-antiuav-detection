"""PR envelopes from saved test matching arrays, with AP50 reproduction checks."""
from pathlib import Path
import hashlib,json
import numpy as np
from ultralytics.utils.metrics import compute_ap
ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
OLD=ROOT/'experiments/aat_temporal_100e_20261017/completed_review_20261017/evaluations/test'
NEW=ROOT/'experiments/tcs_ablation_relaunch_20261017/completed_review_20261018/evaluations/test'
def main():
    grid=np.linspace(0,1,501);rows={}
    for name in ['yolov8n','yolo11n','C1_temporal_uniform','tcs_yolo8n']:
        folder=(NEW if name=='tcs_yolo8n' else OLD)/name;f=folder/'frame_stats.npz';d=np.load(f)
        assert np.all(d['pred_cls']==0) and np.all(d['target_cls']==0)
        order=np.argsort(-d['conf']);tp=d['tp'][order,0].cumsum();nt=len(d['target_cls'])
        recall=tp/(nt+1e-16);precision=tp/np.arange(1,len(tp)+1)
        ap,mpre,mrec=compute_ap(recall,precision)
        audit=json.loads((folder/'audit.json').read_text());assert abs(ap-audit['metrics']['metrics/mAP50(B)'])<1e-9
        rows[name]=dict(AP50=float(ap),recall_grid=grid.tolist(),precision_envelope=np.interp(grid,mrec,mpre).tolist(),
            objects=nt,frames=len(d['frame_stems']),stats_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    out=dict(scope='Fixed completed test checkpoints, all background frames included. Single-class IoU0.5 precision envelope; curves are not new training results.',models=rows)
    (Path(__file__).parent/'pr_data.json').write_text(json.dumps(out,indent=2));print('PR_DATA_VERIFIED')
if __name__=='__main__':main()
