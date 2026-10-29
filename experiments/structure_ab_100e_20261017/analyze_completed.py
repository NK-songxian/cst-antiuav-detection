"""Summarize all three validation runs, matched-FPR diagnostics, and clustered rate intervals."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY=ROOT/'experiments/structure_ab_100e_20261017'
AUDIT=ROOT/'runs/structure_ab_100e_20261017/final_audit/baseline_rect'

def main():
    index=json.loads((ROOT/'data/yolo/attribute_index.json').read_text())['val']
    output={'protocol':'Validation only; same frozen AP evaluator; 9066 frames incl 1247 negatives; matched-FPR thresholds fitted and measured on same validation set: exploratory.',
            'models':{}}
    arrays={}
    reference=None
    for name in ['yolo11n','A_zero_deploy','B_p2_compact']:
        report=json.loads((AUDIT/name/'audit.json').read_text())
        data=np.load(AUDIT/name/'frame_stats.npz')
        stems=data['frame_stems'].tolist()
        if reference is None:reference=stems
        assert reference==stems
        n=len(stems);positive=np.zeros(n,dtype=bool)
        positive[data['target_image_idx']]=True
        score=np.full(n,-np.inf);hit=np.full(n,-np.inf)
        np.maximum.at(score,data['pred_image_idx'],data['conf'])
        np.maximum.at(hit,data['pred_image_idx'],np.where(data['tp'][:,0],data['conf'],-np.inf))
        budget=int((~positive).sum()*.05)
        boundary=np.sort(score[~positive])[::-1][budget]
        threshold=float(np.nextafter(float(boundary),np.inf))
        tc=np.array([bool(index[k]['attr']['TC']) for k in stems])
        row={'metrics':report['metrics'],'protocol':report['protocol'],
             'background_fpr_conf025':report['groups']['ALL']['conf_0.25']['background_frame_fp_rate'],
             'background_fp_frames_conf025':report['groups']['ALL']['conf_0.25']['background_frames_with_fp'],
             'positive_hit_conf025':float(np.mean(hit[positive]>=.25)),
             'matched_fpr05':{'threshold':threshold,'negative_fpr':float(np.mean(score[~positive]>=threshold)),
                 'positive_hit':float(np.mean(hit[positive]>=threshold)),
                 'tc_positive_hit':float(np.mean(hit[positive & tc]>=threshold))},
             'attribute_ap50':{k:report['groups'][k]['metrics/mAP50(B)'] for k in ['TC','SV','DBC','CDB','SIZE_CODE_1_POS_ONLY','SIZE_CODE_2_POS_ONLY','SIZE_CODE_3_POS_ONLY','SIZE_CODE_4_POS_ONLY']},
             'attribute_ap5095':{k:report['groups'][k]['metrics/mAP50-95(B)'] for k in ['TC','SV','DBC','CDB','SIZE_CODE_1_POS_ONLY']},
             'source_sha256':hashlib.sha256((AUDIT/name/'audit.json').read_bytes()).hexdigest()}
        arrays[name]={'positive':positive,'hit025':hit>=.25,'fp025':score>=.25,'hit_at_fpr05':hit>=threshold}
        output['models'][name]=row
    seq=np.array([index[k]['seq'] for k in reference]);names=sorted(set(seq))
    rng=np.random.default_rng(20261017)
    bootstrap=rng.integers(0,len(names),size=(2000,len(names)))
    output['cluster_bootstrap']={'unit':'validation sequence','sequences':len(names),'resamples':2000,
        'interpretation':'Descriptive paired percentile intervals, conditional on the single trained seed and thresholds tuned on this validation set; not AP confidence intervals or independent replication.',
        'comparisons':{}}
    base=arrays['yolo11n'];pos=base['positive']
    for name in ['A_zero_deploy','B_p2_compact']:
        comp={}
        for metric,selected in [('fp025',~pos),('hit025',pos),('hit_at_fpr05',pos)]:
            denom=np.array([np.sum((seq==s)&selected) for s in names])
            a=np.array([np.sum(arrays[name][metric][(seq==s)&selected]) for s in names])
            b=np.array([np.sum(base[metric][(seq==s)&selected]) for s in names])
            d=(a[bootstrap].sum(1)-b[bootstrap].sum(1))/np.maximum(denom[bootstrap].sum(1),1)
            comp[metric]={'difference':float((a.sum()-b.sum())/denom.sum()),'ci95':np.quantile(d,[.025,.975]).tolist()}
        output['cluster_bootstrap']['comparisons'][name]=comp
    (STUDY/'completed_comparison.json').write_text(json.dumps(output,indent=2))
    with (STUDY/'completed_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['model','AP50','AP5095','TC_AP50','tiny_AP50','background_FPR025','positive_hit025','positive_hit_at_FPR05'])
        writer.writeheader()
        for name,row in output['models'].items():
            writer.writerow({'model':name,'AP50':row['metrics']['metrics/mAP50(B)'],'AP5095':row['metrics']['metrics/mAP50-95(B)'],
                'TC_AP50':row['attribute_ap50']['TC'],'tiny_AP50':row['attribute_ap50']['SIZE_CODE_1_POS_ONLY'],
                'background_FPR025':row['background_fpr_conf025'],'positive_hit025':row['positive_hit_conf025'],
                'positive_hit_at_FPR05':row['matched_fpr05']['positive_hit']})
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
