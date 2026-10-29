from pathlib import Path
import json,zipfile,hashlib
R=Path('/root/autodl-tmp/Ryan/cst_antiuav')
O=R/'experiments/protocol_fp_study_20261018/completion_v4.zip'
files={}
for src,dest in [
('experiments/cst_paper_completion_20261018/completion_metrics.json','new_metrics/completion_metrics.json'),
('experiments/cst_paper_completion_20261018/model_costs.json','new_metrics/model_costs.json'),
('experiments/cst_paper_completion_20261018/camera_motion_audit.json','new_metrics/camera_motion_audit.json'),
('experiments/cst_paper_completion_20261018/common_args.json','new_metrics/common_args.json'),
('experiments/cst_paper_completion_20261018/contract.json','new_metrics/contract.json'),
('runs/cst_paper_completion_20261018/queue/status.json','new_metrics/completed_queue_status.json'),
('experiments/protocol_fp_study_20261018/gpu_pipeline_benchmark.json','new_metrics/gpu_pipeline_benchmark.json'),
('experiments/protocol_fp_study_20261018/gpu_profile_status.json','new_metrics/gpu_profile_status.json')]:
 p=R/src
 if p.exists():files[dest]=p
for split in ['validation_audit','test_audit']:
 for p in (R/'runs/cst_paper_completion_20261018'/split).glob('*/*'):
  if p.name in ['audit.json','groups.json','frame_stats.npz']:files['new_metrics/'+split+'/'+p.parent.name+'/'+p.name]=p
for p in (R/'runs/cst_paper_completion_20261018').glob('*/args.yaml'):files['new_metrics/train_args/'+p.parent.name+'.yaml']=p
p=R/'runs/tcs11s_recovery_20261018/tcs_yolo11s/args.yaml'
if p.exists():files['new_metrics/train_args/S_tcs_seed0_adopted.yaml']=p
manifest={name:{'source':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for name,p in files.items()}
with zipfile.ZipFile(O,'w',zipfile.ZIP_DEFLATED) as z:
 for name,p in files.items():z.write(p,name)
 z.writestr('new_metrics/completion_v4_manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'zip':str(O),'files':len(files),'bytes':O.stat().st_size}))
