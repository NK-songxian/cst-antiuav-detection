"""Bounded hardware/software provenance, without environment variables or credentials."""
import json,platform,subprocess,sys,importlib.metadata
from datetime import datetime,timezone
from pathlib import Path
H=Path(__file__).parent
def cmd(args):
    p=subprocess.run(args,capture_output=True,text=True);return dict(argv=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
versions={}
for name in ['numpy','torch','torchvision','ultralytics','opencv-python','scipy','psutil','PyYAML']:
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions[name]=None
d=dict(captured_at=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),packages=versions,gpu=cmd(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader']),cpu=cmd(['lscpu']),disk=cmd(['df','-h',str(H)]),scope='No credential, environment-variable, or shell-history capture')
(H/'environment_snapshot.json').write_text(json.dumps(d,indent=2));print('ENVIRONMENT_SNAPSHOT_SAVED')
