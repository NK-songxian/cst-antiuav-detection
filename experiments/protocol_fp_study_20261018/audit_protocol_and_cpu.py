"""Read-only native annotation audit and CPU input overhead measurement."""
from pathlib import Path
import json,hashlib,time,platform
import numpy as np
import cv2
R=Path('/root/autodl-tmp/Ryan/cst_antiuav');OUT=R/'experiments/protocol_fp_study_20261018';OUT.mkdir(exist_ok=True)
def main():
    cv2.setNumThreads(1);index=json.loads((R/'data/yolo/attribute_index.json').read_text());raw=R/'data/CST-AntiUAV';summary={}
    rawseqs=[]
    for split,items in index.items():
        stats={'sequences':len({x['seq'] for x in items.values()}),'frames':len(items),'present':0,'empty':0,'label_mismatch':0,'source_mismatch':0,'tiny_present_filtered':0,'oob_boxes':0,'dummy_absent_boxes_ignored':0,'attribute_length_mismatch':0};cache={}
        for seq in sorted({x['seq'] for x in items.values()}):
            p=raw/split/seq/'IR_label.json';lab=json.loads(p.read_text());cache[seq]=lab;rawseqs.append(str(p));n=len(lab['exist'])
            stats['attribute_length_mismatch']+=sum(len(lab[k])!=n for k in ['gt','OC','OV','SV','TC','DBC','CDB','FM','SIZE'] if k in lab)
        for stem,x in items.items():
            l=cache[x['seq']];i=x['frame'];e=l['exist'][i];b=l['gt'][i];px,py,w,h=map(float,b[:4]);expected=e==1 and w>1 and h>1
            stats['present']+=expected;stats['empty']+=not expected;stats['tiny_present_filtered']+=e==1 and not expected
            stats['source_mismatch']+=int(e!=x['exist'])+sum(int(l[k][i])!=x['attr'][k] for k in x['attr'])
            labfile=R/f'data/yolo/{split}/labels/{stem}.txt';lines=labfile.read_text().splitlines()
            if expected:
                v=np.array([(px+w/2)/640,(py+h/2)/512,w/640,h/512]);v=np.clip(v,0,1)
                stats['label_mismatch']+=not(len(lines)==1 and int(lines[0].split()[0])==0 and np.max(np.abs(np.array(list(map(float,lines[0].split()[1:])))-v))<=.000001)
                stats['oob_boxes']+=px<0 or py<0 or px+w>640 or py+h>512
            else:
                stats['label_mismatch']+=len(lines)!=0;stats['dummy_absent_boxes_ignored']+=e==0 and w>0 and h>0
        summary[split]=stats
    rawsets={s:{x['seq'] for x in d.values()} for s,d in index.items()};assert not(rawsets['train']&rawsets['val'] or rawsets['train']&rawsets['test'] or rawsets['val']&rawsets['test'])
    for v in summary.values():assert not(v['label_mismatch'] or v['source_mismatch'] or v['attribute_length_mismatch'])
    output={'split_audit':summary,'index_sha256':hashlib.sha256((R/'data/yolo/attribute_index.json').read_bytes()).hexdigest(),'converter_sha256':hashlib.sha256((R/'scripts/prep_cst_yolo.py').read_bytes()).hexdigest(),'coordinate_rule':'Source top-left pixel xywh to normalized centre xywh, class0; exist0 => empty even with dummy coordinates; only exist1&w>1&h>1 yield positive; original frame zero-based metadata index.', 'scope':'Read-only verification; no label changes'}
    (OUT/'protocol_audit.json').write_text(json.dumps(output,indent=2))
    # Decode/resize/input construction are CPU-stage timings, concurrent-training environment recorded.
    stems=sorted(index['test']);data=[]
    for stem in stems[::max(1,len(stems)//64)][:64]:
        p=R/f'data/yolo/test/images/{stem}.jpg';data.append(np.frombuffer(p.read_bytes(),np.uint8))
    timing={k:[] for k in ['decode_ms','current_resize_ms','repeat3_ms','cached_temporal_stack_ms','cached_temporal_total_ms']}
    for it in range(5):
        for encoded in data:
            t=time.perf_counter();img=cv2.imdecode(encoded,cv2.IMREAD_GRAYSCALE);t1=time.perf_counter()
            # Native640x512 -> letterbox to544x672; previous frames assumed already decoded in online cache.
            out=cv2.copyMakeBorder(img,16,16,16,16,cv2.BORDER_CONSTANT,value=114);t2=time.perf_counter();single=np.repeat(out[:,:,None],3,axis=2);t3=time.perf_counter();temp=np.stack([out,out.copy(),out.copy()],axis=2);t4=time.perf_counter()
            if it:
                for key,value in [('decode_ms',t1-t),('current_resize_ms',t2-t1),('repeat3_ms',t3-t2),('cached_temporal_stack_ms',t4-t3),('cached_temporal_total_ms',(t1-t)+(t2-t1)+(t4-t3))]:timing[key].append(value*1000)
    result={'environment':{'platform':platform.platform(),'opencv':cv2.__version__,'threads':1,'co_running_training':True},'n_images':len(data),'measured_iterations':len(timing['decode_ms']),'input':'640x512 grayscale JPEG, letterbox544x672; historical-frame buffers represented by copied decoded arrays; no GPU timing','stage_stats':{k:{'median_ms':float(np.median(v)),'p90_ms':float(np.percentile(v,90))} for k,v in timing.items()},'warning':'CPU-stage benchmark in shared training environment; historical images are reused buffers for operation cost, not detector accuracy. Not Jetson or end-to-end deployment.'}
    (OUT/'cpu_input_benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(output));print(json.dumps(result))
if __name__=='__main__':main()
