"""Gold channel-order, cache protection, augmentation and label invariants."""
import json,random,hashlib,pickle
from pathlib import Path
import numpy as np,torch,yaml
import matched_runtime as rt
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.augment as aug
def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n)
def main():
    result=[]
    x=np.zeros((4,7,3),dtype=np.uint8);x[:,:,0]=11;x[:,:,1]=22;x[:,:,2]=33
    y=rt.current_only_bgr(x);assert np.all(y==33) and np.all(x[:,:,0]==11)
    result.append('BGR current=channel2, output copied; source not mutated')
    r=rt.ROOT;idx=json.loads((r/'data/yolo/attribute_index.json').read_text())
    seq={s:{v['seq'] for v in idx[s].values()} for s in idx}
    assert not seq['train']&seq['val'] and not seq['train']&seq['test'] and not seq['val']&seq['test']
    result.append('Train/val/test sequence sets disjoint')
    common=json.loads((r/'experiments/tcs11s_recovery_20261018/common_args.json').read_text())
    a=get_cfg(overrides=common);a.task='detect';a.mode='train'
    cfg=yaml.safe_load((r/'experiments/aat_temporal_100e_20261017/cst_temporal.yaml').read_text())
    aug.RandomHSV=rt.ti.SharedBrightness
    ds={mode:rt.make_dataset(a,cfg['train'],256,cfg,'train',32,mode) for mode in ['current','temporal']}
    assert ds['current'].im_files==ds['temporal'].im_files and len(ds['current'])==25612
    for i in [0,13,128,63]:
        seed(20261018+i);u=ds['current'][i]
        seed(20261018+i);v=ds['temporal'][i]
        assert torch.equal(u['img'][0],u['img'][1]) and torch.equal(u['img'][1],u['img'][2])
        assert torch.equal(u['img'][0],v['img'][0]),i
        assert torch.equal(u['bboxes'],v['bboxes']) and torch.equal(u['cls'],v['cls']),i
    result.append('Four real augmented samples: current pixels exactly match temporal current channel; GT identical')
    for mode in ['current','temporal']:
        restored=pickle.loads(pickle.dumps(ds[mode]));assert restored.input_mode==mode
    result.append('Input mode survives dataset serialization to worker processes')
    sample=ds['current'].load_image(0)[0];cached=ds['current'].ims[0]
    assert np.array_equal(sample[:,:,0],cached[:,:,2])
    result.append('Cached temporal source preserved while current-only image returned')
    out=Path(__file__).parent/'preflight.json';out.write_text(json.dumps(dict(status='PASS',checks=result),indent=2))
    print('PREFLIGHT_PASS',json.dumps(result),flush=True)
if __name__=='__main__':main()
