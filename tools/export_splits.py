"""Export official sequence/sample split lists from the converter index."""
from pathlib import Path
import argparse,json,hashlib

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();raw=a.index.read_bytes();idx=json.loads(raw);a.out.mkdir(parents=True,exist_ok=True)
    summary={};seen=set()
    for split in ['train','val','test']:
        rows=idx[split];seqs=sorted({v['seq'] for v in rows.values()})
        if seen.intersection(seqs):raise ValueError('A sequence occurs in multiple splits')
        seen.update(seqs)
        (a.out/(split+'_sequences.txt')).write_text('\n'.join(seqs)+'\n',encoding='utf-8')
        (a.out/(split+'_samples.txt')).write_text('\n'.join(split+'/images/'+s+'.jpg' for s in sorted(rows))+'\n',encoding='utf-8')
        summary[split]={'sequences':len(seqs),'samples':len(rows),'target_present':sum(v['exist']==1 for v in rows.values()),'target_absent':sum(v['exist']==0 for v in rows.values())}
    result={'index_sha256':hashlib.sha256(raw).hexdigest(),'splits':summary,'sequence_overlap':False}
    (a.out/'split_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
