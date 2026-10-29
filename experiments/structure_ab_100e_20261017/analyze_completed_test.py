"""One-time held-out comparison; reuse thresholds selected on validation only."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY=ROOT/'experiments/structure_ab_100e_20261017'
AUDIT=ROOT/'runs/structure_ab_100e_20261017/final_audit/baseline_rect_test'
validation=json.loads((STUDY/'completed_comparison.json').read_text())
report={'protocol':'Held-out test; three fixed checkpoints selected by validation AP50-95; no test threshold fitting or checkpoint selection.',
        'models':{}}
for name in ['yolo11n','A_zero_deploy','B_p2_compact']:
    a=json.loads((AUDIT/name/'audit.json').read_text())
    d=np.load(AUDIT/name/'frame_stats.npz')
    n=len(d['frame_stems']);pos=np.zeros(n,dtype=bool);pos[d['target_image_idx']]=True
    score=np.full(n,-np.inf);hit=np.full(n,-np.inf)
    np.maximum.at(score,d['pred_image_idx'],d['conf'])
    np.maximum.at(hit,d['pred_image_idx'],np.where(d['tp'][:,0],d['conf'],-np.inf))
    threshold=validation['models'][name]['matched_fpr05']['threshold']
    row={'metrics':a['metrics'],'protocol':a['protocol'],
         'background_fpr_conf025':a['groups']['ALL']['conf_0.25']['background_frame_fp_rate'],
         'background_fp_frames_conf025':a['groups']['ALL']['conf_0.25']['background_frames_with_fp'],
         'positive_hit_conf025':a['groups']['ALL']['conf_0.25']['positive_frame_hit_rate_iou50'],
         'attribute_ap50':{k:a['groups'][k]['metrics/mAP50(B)'] for k in ['TC','SV','DBC','CDB','SIZE_CODE_1_POS_ONLY','SIZE_CODE_2_POS_ONLY','SIZE_CODE_3_POS_ONLY','SIZE_CODE_4_POS_ONLY']},
         'attribute_ap5095':{k:a['groups'][k]['metrics/mAP50-95(B)'] for k in ['TC','SV','DBC','CDB','SIZE_CODE_1_POS_ONLY']},
         'validation_threshold_replayed':{'threshold':threshold,'test_background_fpr':float(np.mean(score[~pos]>=threshold)),
                                        'test_positive_hit':float(np.mean(hit[pos]>=threshold)),
                                        'note':'These test FPRs are not constrained to 5%; no retuning on test.'}}
    report['models'][name]=row
(STUDY/'completed_test_comparison.json').write_text(json.dumps(report,indent=2))
with (STUDY/'completed_test_comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['model','AP50','AP5095','TC_AP50','tiny_AP50','background_FPR025','positive_hit025'])
    writer.writeheader()
    for name,row in report['models'].items():
        writer.writerow({'model':name,'AP50':row['metrics']['metrics/mAP50(B)'],'AP5095':row['metrics']['metrics/mAP50-95(B)'],
            'TC_AP50':row['attribute_ap50']['TC'],'tiny_AP50':row['attribute_ap50']['SIZE_CODE_1_POS_ONLY'],
            'background_FPR025':row['background_fpr_conf025'],'positive_hit025':row['positive_hit_conf025']})
print(json.dumps(report,indent=2))
