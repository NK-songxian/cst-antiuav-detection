"""Update only the marked results section; never insert forecasts or select a best seed."""
import argparse,csv,hashlib,json,statistics,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY=ROOT/'experiments/cst_paper_completion_20261018'
RUNS=ROOT/'runs/cst_paper_completion_20261018'
START='<!-- COMPLETION_RESULTS_START -->';END='<!-- COMPLETION_RESULTS_END -->'
K50,K95='metrics/mAP50(B)','metrics/mAP50-95(B)'
def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None
def percent(v):return '待完成' if v is None else f'{v*100:.2f}'
def refresh(outline):
    state=read(RUNS/'queue/status.json');contract=read(STUDY/'contract.json')
    if state is None or contract is None:return False
    spec={j['name']:j for j in contract['models']};rows=[]
    for j in state['jobs']:
        name=j['name'];job=spec[name]
        meta=read(Path(contract['adopted_training']['manifest'])) if job.get('adopted') else read(STUDY/'jobs'/name/'manifest.json')
        va=read(RUNS/'validation_audit'/name/'audit.json');ta=read(RUNS/'test_audit'/name/'audit.json')
        if va is not None:assert va['protocol']['frames']==9066 and va['protocol']['includes_negatives']
        if ta is not None:assert ta['protocol']['frames']==14446 and ta['groups']['ALL']['negative_frames']==2592
        if va is not None and ta is not None:assert va['weights_sha256']==ta['weights_sha256']
        rows.append(dict(name=name,mode=job['input_mode'],seed=job['seed'],arch=job['arch'],state=j['state'],stage=j['stage'],
            epoch=meta.get('latest_epoch',meta.get('last_epoch')) if meta else None,
            val_AP50=va['metrics'][K50] if va else None,test_AP50=ta['metrics'][K50] if ta else None,
            test_AP5095=ta['metrics'][K95] if ta else None,
            test_BG_FP025=ta['groups']['ALL']['conf_0.25']['background_frame_fp_rate'] if ta else None))
    byname={r['name']:r for r in rows};pairs=[]
    for seed in [0,1,2]:
        a,b=byname[f'N_current_seed{seed}'],byname[f'N_tcs_seed{seed}']
        if a['test_AP50'] is not None and b['test_AP50'] is not None:
            pairs.append(dict(seed=seed,current_AP50=a['test_AP50'],tcs_AP50=b['test_AP50'],difference_pp=100*(b['test_AP50']-a['test_AP50'])))
    stats=dict(paired_seeds_complete=len(pairs),paired_seeds_planned=3,seed_pairs=pairs)
    if pairs:
        for key in ['current_AP50','tcs_AP50','difference_pp']:
            x=[p[key] for p in pairs];stats[key]=dict(mean=statistics.mean(x),sd=statistics.stdev(x) if len(x)>1 else None,n=len(x))
    payload=dict(as_of=datetime.now(timezone.utc).isoformat(),queue_status=state['status'],rows=rows,paired_summary=stats,
        interpretation='Only completed fixed-checkpoint audits enter numeric cells. Uncompleted values remain missing. Paired means use the intersection of completed seeds; no best-seed selection.')
    (STUDY/'completion_metrics.json').write_text(json.dumps(payload,indent=2))
    with (STUDY/'completion_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['',f"更新时间：{payload['as_of']}。队列状态：{state['status']}。以下为补实验实时状态，AP均为百分比。",'',
      '| 条件 | 状态 / 阶段 | 轮数 | val AP50 | test AP50 | test AP50–95 | test BG-FP@0.25 |',
      '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('|'+ '|'.join([r['name'],r['state']+' / '+r['stage'],str(r['epoch']) if r['epoch'] is not None else '—',
      percent(r['val_AP50']),percent(r['test_AP50']),percent(r['test_AP5095']),percent(r['test_BG_FP025'])])+'|')
    lines += ['',f"YOLO11n 已完成配对种子：{len(pairs)}/3。只有对应两组均完成test，才计入配对汇总。"]
    if len(pairs)==3:
        for key,label in [('current_AP50','单帧 AP50'),('tcs_AP50','TCS AP50')]:
            x=stats[key];lines.append(f"- {label}：{x['mean']*100:.2f}% ± {x['sd']*100:.2f} 个百分点（训练种子标准差，n=3）。")
        x=stats['difference_pp'];lines.append(f"- 配对差值：{x['mean']:+.2f} ± {x['sd']:.2f} 点；不据此编造显著性或选择最高种子。")
    else:lines += ['三种子尚未全部完成，不将部分均值写成最终结果。']
    lines += ['','原历史单次表保留来源，定稿时用这套同数据流配对结果统一更新；负结果和失败也保留。','']
    if outline.exists():
        text=outline.read_text();assert text.count(START)==1 and text.count(END)==1
        head,tail=text.split(START);_,tail=tail.split(END)
        updated=head+START+'\n'+'\n'.join(lines)+END+tail
        if updated!=text:
            tmp=outline.with_suffix('.refresh.tmp');tmp.write_text(updated);tmp.replace(outline)
            with (STUDY/'outline_refresh_log.jsonl').open('a') as f:
                f.write(json.dumps(dict(at=payload['as_of'],outline_sha256=hashlib.sha256(updated.encode()).hexdigest(),paired_seeds_complete=len(pairs)))+'\n')
    return state['status'] in ['completed','needs_attention']
def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    outline=ROOT/'CST_C1_论文大纲_20261017.md'
    while True:
        finished=refresh(outline)
        if a.once or finished:break
        time.sleep(30)
if __name__=='__main__':main()
