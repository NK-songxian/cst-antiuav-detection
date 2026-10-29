"""Reanalyse frozen predictions; no training, threshold tuning or DOCX edits."""
import ast, csv, hashlib, json, types,os,sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).parent
R=Path(os.environ.get('CST_ANALYSIS_ROOT',str(HERE.parent)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_ap():
    # Load only the NumPy metric functions, avoiding this workstation's incomplete torch install.
    source=HERE/'ultralytics_metrics_v8_4_177.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('smooth','compute_ap','ap_per_class')]
    assert len(nodes)==3
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    scope=dict(np=np,Path=Path,checks=types.SimpleNamespace(check_version=lambda *a,**k:int(np.__version__.split('.')[0])>=2))
    exec(compile(ast.fix_missing_locations(module),str(source),'exec'),scope)
    return scope['ap_per_class'],dict(path=str(source),sha256=sha(source))

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    apfn,metric_source=load_ap()
    idx_path=R/'new_metrics/attribute_index.json'
    index=json.loads(idx_path.read_text(encoding='utf-8'))['test']
    fa=json.loads((R/'false_alarm_analysis.json').read_text(encoding='utf-8'))
    events=json.loads((R/'reappearance_analysis.json').read_text(encoding='utf-8'))
    ap_rows=[];cal_rows=[];proof=[];regression=[]
    for seed in (0,1,2):
        for mode,cn in [('current','单帧'),('tcs','三帧')]:
            name=f'N_{mode}_seed{seed}';label=f'{"单帧" if mode=="current" else "时序"}seed{seed}'
            path=R/'new_metrics/test_audit'/name/'frame_stats.npz'
            d=np.load(path);p=fa['models'][label]['provenance']
            assert sha(path)==p['test_stats_sha256']
            val_path=R/'new_metrics/validation_audit'/name/'frame_stats.npz'
            assert sha(val_path)==p['val_stats_sha256']
            stems=d['frame_stems'];meta=[index[str(s)] for s in stems]
            pos=np.array([bool(x['exist']) for x in meta]);assert pos.sum()==11854
            saved=json.loads((path.parent/'groups.json').read_text(encoding='utf-8'))
            def ap(mask):
                pm=mask[d['pred_image_idx']];tm=mask[d['target_image_idx']]
                result=apfn(d['tp'][pm],d['conf'][pm],d['pred_cls'][pm],d['target_cls'][tm],plot=False)
                return result[5]
            # Regression against every saved ALL and TC_POS_ONLY metric, both IoU columns.
            all_ap=ap(np.ones(len(stems),bool))
            for key,value in [('metrics/mAP50(B)',all_ap[:,0].mean()),('metrics/mAP50-95(B)',all_ap.mean())]:
                delta=float(value-saved['ALL'][key]);regression.append(dict(model=name,group='ALL',metric=key,error_pp=delta*100))
                assert abs(delta)<5e-5,(name,key,value,saved['ALL'][key])
            for attr in ['TC','SV','DBC','CDB']:
                mask=np.array([x['attr'].get(attr)==1 for x in meta])
                a=ap(mask&pos)
                if attr=='TC':
                    for key,value in [('metrics/mAP50(B)',a[:,0].mean()),('metrics/mAP50-95(B)',a.mean())]:
                        delta=float(value-saved['TC_POS_ONLY'][key]);regression.append(dict(model=name,group='TC_POS_ONLY',metric=key,error_pp=delta*100))
                        assert abs(delta)<5e-5,(name,'TC',key,value,saved['TC_POS_ONLY'][key])
                ap_rows.append(dict(model=name,input=cn,seed=seed,attribute=attr,positive_frames=int((mask&pos).sum()),negative_frames=int((mask&~pos).sum()),AP50=float(a[:,0].mean()*100),AP5095=float(a.mean()*100)))
            c=fa['models'][label]['calibration'][2];assert c['validation_budget']==.05
            v=np.load(val_path);vpos=np.bincount(v['target_image_idx'],minlength=len(v['frame_stems']))>0
            vfp=np.bincount(v['pred_image_idx'][v['conf']>=c['threshold']],minlength=len(vpos))>0
            assert int((vfp&~vpos).sum())==c['val']['fp_frames']
            hit=np.bincount(d['pred_image_idx'][(d['conf']>=c['threshold'])&d['tp'][:,0]],minlength=len(stems))>0
            lookup={str(s):i for i,s in enumerate(stems)}
            h=[[hit[lookup[s]] for s in e['stems']] for e in events['events']]
            r1=sum(int(x[0]) for x in h);r3=sum(int(any(x)) for x in h)
            assert len(h)==178 and r1==events['models'][label]['first_sample_hits'] and r3==events['models'][label]['first3_hits']
            cal_rows.append(dict(input=cn,seed=seed,tau=c['threshold'],val_fp=c['val']['fp_frames'],val_bg=1247,val_BG_FP=c['val']['bg_fpr']*100,test_fp=c['test']['fp_frames'],test_bg=2592,test_BG_FP=c['test']['bg_fpr']*100,R1=r1/178*100,R3=r3/178*100))
            proof.append(dict(model=name,test_stats_sha256=sha(path),val_stats_sha256=sha(val_path)))
    summaries=[]
    for attr in ['TC','SV','DBC','CDB']:
        entry={'attribute':attr}
        for mode in ['单帧','三帧']:
            rows=[x for x in ap_rows if x['attribute']==attr and x['input']==mode]
            entry[mode]={k:{'mean':float(np.mean([x[k] for x in rows])),'sd':float(np.std([x[k] for x in rows],ddof=1))} for k in ['AP50','AP5095']}
            entry['positive_frames']=rows[0]['positive_frames'];entry['negative_frames']=rows[0]['negative_frames']
        summaries.append(entry)
    result=dict(status='PASS_BOUNDED_NUMERICAL_REGRESSION',no_training=True,metric_source=metric_source,numpy_version=np.__version__,index_sha256=sha(idx_path),regression=regression,max_regression_error_pp=max(abs(x['error_pp']) for x in regression),AP_scope='positive frames only for all four attributes; attributes overlap; SD over three seeds; local cross-platform tied-confidence sorting can slightly change AP',sources=proof,attributes=summaries,per_seed_AP=ap_rows,calibration=cal_rows)
    (HERE/'v2_reanalysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for filename,rows in [('属性AP_逐种子.csv',ap_rows),('工作点_R1_R3_六行.csv',cal_rows)]:
        with (HERE/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summaries,ensure_ascii=False,indent=2));print('REGRESSION_PASS')

if __name__=='__main__':main()
