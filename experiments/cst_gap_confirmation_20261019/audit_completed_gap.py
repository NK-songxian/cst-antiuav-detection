"""Verify every run and archive bounded raw evidence; no training or model selection."""
import csv,hashlib,json,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from ultralytics.utils.metrics import ap_per_class
H=Path(__file__).parent
ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    c=json.loads((H/'contract.json').read_text());q=json.loads((H/'queue/status.json').read_text());summary=json.loads((H/'results/summary.json').read_text())
    assert q['status']=='completed' and len(q['jobs'])==6
    assert all(x['state']=='completed' and x['stage']=='test' and x['returncode']==0 for x in q['jobs'])
    assert summary['contract_sha256']==sha(H/'contract.json')
    for p,h in c['code_sha256'].items():assert sha(p)==h,p
    proof={x['name']:x for x in summary['provenance']};files={};report=[];ap_checks=[]
    def add(p,rel):
        assert p.is_file(),p
        files[rel]=p
    for row in summary['rows']:
        name=row['name'];far=row['far'];base=ROOT/'runs'/('cst_paper_completion_20261018' if far==10 else 'cst_gap_confirmation_20261019')
        run=base/name;checkpoint=run/'weights/best.pt'
        assert sha(checkpoint)==proof[name]['weights_sha256']
        assert checkpoint.is_file()
        add(run/'args.yaml',f'train/{name}/args.yaml');add(run/'results.csv',f'train/{name}/results.csv')
        epochs=list(csv.DictReader((run/'results.csv').open()));assert epochs and len(epochs)<=100
        if far!=10:
            mp=H/'jobs'/name/'manifest.json';m=json.loads(mp.read_text())
            assert m['status']=='trained' and m['actual_batch']==256 and m['actual_train_workers']==8
            assert m['best_sha256']==sha(checkpoint)
            assert m['contract_sha256']==sha(H/'contract.json')
            assert m['job']['seed']==row['seed'] and m['job']['far']==far
            add(mp,f'train/{name}/manifest.json')
            for stage in ['train','val','test']:add(H/'queue/logs'/f'{name}_{stage}.log',f'logs/{name}_{stage}.log')
        for split,folder in [('val','validation_audit'),('test','test_audit')]:
            dest=base/folder/name;ap=dest/'audit.json';d=json.loads(ap.read_text());stats=dest/'frame_stats.npz'
            prefix='val' if split=='val' else 'test'
            assert sha(ap)==proof[name][prefix+'_audit_sha256']
            assert sha(stats)==proof[name][prefix+'_stats_sha256']
            assert d['weights_sha256']==sha(checkpoint)
            s=np.load(stats);n=len(s['frame_stems']);positive=np.bincount(s['target_image_idx'],minlength=n)>0
            assert (n,int(positive.sum()))==((9066,7819) if split=='val' else (14446,11854))
            assert s['tp'].shape[1]==10 and np.all(s['target_cls']==0) and np.all(s['pred_cls']==0)
            metric=ap_per_class(s['tp'],s['conf'],s['pred_cls'],s['target_cls'],plot=False)[5]
            assert abs(metric[:,0].mean()-d['metrics']['metrics/mAP50(B)'])<1e-12
            assert abs(metric.mean()-d['metrics']['metrics/mAP50-95(B)'])<1e-12
            pred=np.bincount(s['pred_image_idx'][s['conf']>=.25],minlength=n)>0
            bg_fp=int((pred&~positive).sum());g=d['groups']['ALL']['conf_0.25']
            assert bg_fp==g['background_frames_with_fp']
            if split=='test':assert abs(row['test_BG_FP025']-100*bg_fp/(~positive).sum())<1e-10
            ap_checks.append(dict(model=name,split=split,frames=n,positive_frames=int(positive.sum()),background_fp025=bg_fp,AP50=float(metric[:,0].mean()),AP5095=float(metric.mean()),replay_max_error=0))
            for f in ['audit.json','groups.json','frame_stats.npz']:add(dest/f,f'evaluations/{split}/{name}/{f}')
        report.append(dict(name=name,far=far,seed=row['seed'],training_epochs=len(epochs),best_pt_sha256=sha(checkpoint),best_pt_path=str(checkpoint),best_pt_bytes=checkpoint.stat().st_size))
    for f in ['contract.json','experiment_plan.json','environment_snapshot.json','bootstrap.log','attribute_lineage_full_audit.json','collect_gap_results.py','audit_completed_gap.py','train_matched.py','matched_runtime.py','gap_protocol.py','evaluate_gap.py','queue_gap.py']:
        add(H/f,'study/'+f)
    add(H/'queue/status.json','study/completed_queue_status.json')
    for f in ['summary.json','all9_runs.csv','补实验结果.md']:add(H/'results'/f,'results/'+f)
    source_manifest=[dict(archive_path=rel,source=str(p),bytes=p.stat().st_size,sha256=sha(p)) for rel,p in sorted(files.items())]
    out=dict(status='PASS_ALL9_REPLAYED',captured_at=datetime.now(timezone.utc).isoformat(),new_completed_runs=6,reused_runs=3,models=report,replay_checks=ap_checks,contract_sha256=sha(H/'contract.json'),summary_sha256=sha(H/'results/summary.json'),files=source_manifest,scope='Full-set AP and BG-FP replay for all 9 models on val and test; data/code/checkpoint identity; no independent retraining claim.')
    (H/'completion_audit.json').write_text(json.dumps(out,indent=2))
    archive=H/'A5_completed_evidence_20261020.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for rel,p in sorted(files.items()):z.write(p,rel)
        z.write(H/'completion_audit.json','completion_audit.json')
    (H/'completion_archive_manifest.json').write_text(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),audit_sha256=sha(H/'completion_audit.json')),indent=2))
    print('COMPLETION_AUDIT_PASS',len(report),'models',len(ap_checks),'split replays','archive_bytes',archive.stat().st_size)
    print('EPOCHS',[(x['name'],x['training_epochs']) for x in report])
if __name__=='__main__':main()
