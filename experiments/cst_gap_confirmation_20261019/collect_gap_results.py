"""Report all frozen candidates once both queues complete; never select or launch training."""
import csv,hashlib,json,math,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.stats import t
H=Path(__file__).parent
ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def main():
    status=H/'queue/status.json'
    while not status.exists():time.sleep(15)
    while True:
        q=json.loads(status.read_text())
        if q['status']=='completed':break
        if q['status']=='failed_stop':
            (H/'results_not_complete.json').write_text(json.dumps(dict(status='INCOMPLETE',captured_at=now(),queue_sha256=sha(status),note='Queue stopped after failure. No complete comparison or success conclusion.'),indent=2));return
        time.sleep(30)
    c=json.loads((H/'contract.json').read_text());rows=[];proof=[]
    for far in [10,15,20]:
        base=ROOT/'runs'/('cst_paper_completion_20261018' if far==10 else 'cst_gap_confirmation_20261019')
        for seed in [0,1,2]:
            name=f'N_tcs_seed{seed}' if far==10 else f'N_gap{far}_seed{seed}'
            testp=base/'test_audit'/name/'audit.json';valp=base/'validation_audit'/name/'audit.json'
            test=json.loads(testp.read_text());val=json.loads(valp.read_text())
            assert test['weights_sha256']==val['weights_sha256']
            group=test['groups']['ALL']['conf_0.25']
            rows.append(dict(far=far,near=5,seed=seed,name=name,val_AP50=val['metrics']['metrics/mAP50(B)']*100,val_AP5095=val['metrics']['metrics/mAP50-95(B)']*100,test_AP50=test['metrics']['metrics/mAP50(B)']*100,test_AP5095=test['metrics']['metrics/mAP50-95(B)']*100,test_BG_FP025=group['background_frame_fp_rate']*100))
            proof.append(dict(name=name,weights_sha256=test['weights_sha256'],test_audit_sha256=sha(testp),val_audit_sha256=sha(valp),test_stats_sha256=sha(testp.parent/'frame_stats.npz'),val_stats_sha256=sha(valp.parent/'frame_stats.npz')))
    pairs=[]
    for far in [15,20]:
        a=np.array([next(x['test_AP50'] for x in rows if x['far']==10 and x['seed']==s) for s in [0,1,2]])
        b=np.array([next(x['test_AP50'] for x in rows if x['far']==far and x['seed']==s) for s in [0,1,2]])
        delta=a-b;mean=float(delta.mean());sd=float(delta.std(ddof=1));se=sd/math.sqrt(3)
        pairs.append(dict(comparison=f'gap10-minus-gap{far}',paired_differences_pp=delta.tolist(),mean_pp=mean,sd_pp=sd,CI95=[mean-t.ppf(.975,2)*se,mean+t.ppf(.975,2)*se],CI975_Bonferroni_two_comparisons=[mean-t.ppf(.9875,2)*se,mean+t.ppf(.9875,2)*se],caution='Exploratory paired-t intervals, n=3 seeds; normality not established, finite candidates only. Nonsignificance does not establish equivalence.'))
    summaries=[]
    for far in [10,15,20]:
        rs=[x for x in rows if x['far']==far]
        summaries.append(dict(far=far,**{k:dict(mean=float(np.mean([x[k] for x in rs])),sd=float(np.std([x[k] for x in rs],ddof=1))) for k in ['val_AP50','val_AP5095','test_AP50','test_AP5095','test_BG_FP025']}))
    dest=H/'results';dest.mkdir(exist_ok=False)
    with (dest/'all9_runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=dict(status='COMPLETED_ALL9',created_at=now(),contract_sha256=sha(H/'contract.json'),collector_sha256=sha(__file__),reference_runs=3,new_runs=6,rows=rows,summaries=summaries,paired_comparisons=pairs,provenance=proof,selection='Each checkpoint selected by validation AP50-95. No test-directed tuning or extra seeds.',scope='Finite candidate sensitivity comparison, not global optimality or mechanism identification')
    (dest/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# 统一图像组帧间距补实验','', '全部新增6次训练与验证/测试审计完成；保留全部候选与种子。','', '| 原始帧偏移 | 验证mAP50–95 | 测试mAP50 | 测试mAP50–95 | 测试BG-FP@0.25 |','|---|---:|---:|---:|---:|']
    for x in summaries:
        values=['%.2f±%.2f'%(x[k]['mean'],x[k]['sd']) for k in ['val_AP5095','test_AP50','test_AP5095','test_BG_FP025']]
        lines.append('| 0/5/%d | %s |'%(x['far'],' | '.join(values)))
    lines+=['','单位为%，均值±样本SD，三种子；配对差及探索性区间见summary.json。不据此宣称全局最优或不显著即等效。']
    (dest/'补实验结果.md').write_text('\n'.join(lines),encoding='utf-8');print('ALL9_RESULTS_SAVED',flush=True)
if __name__=='__main__':main()
