"""Wait for the owned training queue, then measure batch1 stages on the real GPU."""
from pathlib import Path
import json,time,subprocess,hashlib,os
R=Path('/root/autodl-tmp/Ryan/cst_antiuav');OUT=R/'experiments/protocol_fp_study_20261018';OUT.mkdir(exist_ok=True)
STATE=OUT/'gpu_profile_status.json'
def status(s,**k):STATE.write_text(json.dumps(dict(status=s,pid=os.getpid(),updated_at=time.time(),**k),indent=2))
def main():
    q=R/'runs/cst_paper_completion_20261018/queue/status.json'
    status('waiting_for_training_queue')
    while True:
        st=json.loads(q.read_text())
        if st['status']=='needs_attention':status('waiting_queue_needs_attention');time.sleep(30);continue
        owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if st['status']=='completed' and not owners:break
        time.sleep(30)
    status('profiling')
    import torch,cv2,numpy as np
    from ultralytics import YOLO
    from ultralytics.utils.nms import non_max_suppression
    from torch.profiler import profile,ProfilerActivity,record_function
    torch.set_num_threads(2);cv2.setNumThreads(1)
    index=json.loads((R/'data/yolo/attribute_index.json').read_text())['val'];samples=[]
    lookup={(x['seq'],x['frame']):s for s,x in index.items()}
    for stem in sorted(index)[::max(1,len(index)//32)][:32]:
        x=index[stem];near=lookup.get((x['seq'],x['frame']-5),stem);far=lookup.get((x['seq'],x['frame']-10),near)
        p=R/f'data/yolo/val/images/{stem}.jpg';b=p.read_bytes()
        a=cv2.imread(str(R/f'data/yolo/val/images/{near}.jpg'),cv2.IMREAD_GRAYSCALE)
        c=cv2.imread(str(R/f'data/yolo/val/images/{far}.jpg'),cv2.IMREAD_GRAYSCALE)
        samples.append((np.frombuffer(b,np.uint8),a,c))
    result={'environment':{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cv2':cv2.__version__},'protocol':'Batch1 FP16, 20 warmups, 128 iterations; current JPEG encoded in RAM, histories predecoded in cache, no disk IO; letterbox544x672; NMS confidence.25 IoU.7 maxDet300; synchronous wall-clock stages; separate profiler trace incurs instrumentation overhead.','models':{}}
    for name,folder in [('current','N_current_seed0'),('temporal','N_tcs_seed0')]:
        wp=R/f'runs/cst_paper_completion_20261018/{folder}/weights/best.pt';net=YOLO(str(wp)).model.to('cuda').eval().half();net.fuse(verbose=False);dur=[]
        def step(sample,measure=False):
            encoded,near,far=sample;t0=time.perf_counter()
            with record_function('decode_current_frame'):cur=cv2.imdecode(encoded,cv2.IMREAD_GRAYSCALE)
            t1=time.perf_counter()
            with record_function('assemble_and_letterbox'):
                rgb=np.repeat(cur[:,:,None],3,2) if name=='current' else np.stack([cur,near,far],2)
                rgb=cv2.copyMakeBorder(rgb,16,16,16,16,cv2.BORDER_CONSTANT,value=(114,114,114));array=np.ascontiguousarray(rgb.transpose(2,0,1))[None]
            t2=time.perf_counter()
            with record_function('host_to_device_and_normalize'):x=torch.from_numpy(array).to('cuda').half()/255;torch.cuda.synchronize()
            t3=time.perf_counter()
            with record_function('network_forward'):y=net(x);torch.cuda.synchronize()
            t4=time.perf_counter()
            with record_function('nms'):d=non_max_suppression(y,.25,.7,max_det=300,nc=1);torch.cuda.synchronize()
            t5=time.perf_counter()
            if measure:dur.append([(t1-t0)*1000,(t2-t1)*1000,(t3-t2)*1000,(t4-t3)*1000,(t5-t4)*1000,(t5-t0)*1000])
        with torch.inference_mode():
            for i in range(20):step(samples[i%len(samples)])
            torch.cuda.reset_peak_memory_stats()
            for i in range(128):step(samples[i%len(samples)],True)
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
                for i in range(6):step(samples[i])
            prof.export_chrome_trace(str(OUT/f'{name}_pipeline_trace.json'))
        cols=['decode_ms','assemble_pad_ms','transfer_normalize_ms','forward_ms','nms_ms','total_ms'];a=np.array(dur)
        result['models'][name]={'weights_sha256':hashlib.sha256(wp.read_bytes()).hexdigest(),'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'n':len(dur),'stages':{k:{'median_ms':float(np.median(a[:,j])),'p90_ms':float(np.percentile(a[:,j],90))} for j,k in enumerate(cols)},'samples_ms':dur}
        del net
        torch.cuda.empty_cache()
    (OUT/'gpu_pipeline_benchmark.json').write_text(json.dumps(result,indent=2));status('completed',result=str(OUT/'gpu_pipeline_benchmark.json'))
if __name__=='__main__':
    try:main()
    except Exception as e:status('failed',error=repr(e));raise
