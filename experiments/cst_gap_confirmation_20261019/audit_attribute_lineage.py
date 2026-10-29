"""Compare every retained frame attribute against released IR_label arrays."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
H=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ip=ROOT/'data/yolo/attribute_index.json';index=json.loads(ip.read_text());sources=[];counts={}
for split,records in index.items():
    annotations={}
    for seq in sorted({r['seq'] for r in records.values()}):
        p=ROOT/'data/CST-AntiUAV'/split/seq/'IR_label.json';a=json.loads(p.read_text())
        n=len(a['exist'])
        assert all(len(a[k])==n for k in ['gt','OC','OV','SV','TC','DBC','CDB'])
        for k in ['FM','SIZE']:
            if k in a:assert len(a[k])==n
        annotations[seq]=a;sources.append(dict(split=split,seq=seq,path=str(p),frames=n,sha256=sha(p)))
    field_count=0
    for stem,r in records.items():
        a=annotations[r['seq']];i=r['frame']
        assert r['exist']==int(a['exist'][i]),(split,stem,'exist')
        for k,value in r['attr'].items():
            assert value==int(a[k][i]),(split,stem,k);field_count+=1
    counts[split]=dict(sequences=len(annotations),retained_frames=len(records),attribute_values_checked=field_count)
out=dict(status='PASS',captured_at=datetime.now(timezone.utc).isoformat(),index_sha256=sha(ip),script_sha256=sha(__file__),counts=counts,sources=sources,meaning='All indexed attribute values and existence states match released IR_label.json at original frame indices. No sequence summaries or geometric proxies used for this index.',limits='Validates provenance/alignment, not human label accuracy.')
(H/'attribute_lineage_full_audit.json').write_text(json.dumps(out,indent=2));print('FULL_ATTRIBUTE_LINEAGE_PASS',counts)
