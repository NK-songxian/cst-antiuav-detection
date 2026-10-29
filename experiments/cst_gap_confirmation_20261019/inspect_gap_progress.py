import hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
H=Path(__file__).parent
q=json.loads((H/'queue/status.json').read_text())
jobs=[]
for j in q['jobs']:
    row={k:j.get(k) for k in ['lane','name','seed','stage','state','pid','returncode']}
    p=H/'jobs'/j['name']/'manifest.json'
    if p.exists():
        m=json.loads(p.read_text());row['manifest']={k:m.get(k) for k in ['status','actual_batch','actual_train_workers','actual_val_workers','actual_val_batch','actual_optimizer','latest_epoch','peak_reserved_bytes','error']}
        row['manifest_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    p=H/'queue/logs'/f"{j['name']}_{j['stage']}.log"
    if p.exists():
        text=p.read_text(errors='replace');row['log_tail']=text.splitlines()[-5:];row['log_bytes']=p.stat().st_size
    jobs.append(row)
out=dict(captured_at=datetime.now(timezone.utc).isoformat(),queue_status=q['status'],scheduler_pid=q['scheduler_pid'],max_parallel=q['max_parallel'],active_processes=q['active_processes'],jobs=jobs,gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout,collector_log=(H/'collect_gap_results.py.log').read_text(errors='replace')[-1000:])
(H/'progress_snapshot.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
