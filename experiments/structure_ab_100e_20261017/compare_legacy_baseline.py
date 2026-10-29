"""Read-only checkpoint tensor comparison for the validation/test audit."""
import json
from pathlib import Path
import torch

root = Path('/root/autodl-tmp/Ryan/cst_antiuav')
old = Path(json.loads((root/'baseline_reset_latest.json').read_text())['old_runs'])
paths = [old/'baseline_yolo11n/weights/best.pt',
         root/'runs/baselines_100e_20261017/yolo11n/weights/best.pt']
states = []
for path in paths:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = ckpt.get('ema') or ckpt['model']
    states.append(model.state_dict())
keys_match = states[0].keys() == states[1].keys()
different = []
if keys_match:
    for key in states[0]:
        a, b = states[0][key], states[1][key]
        if a.shape != b.shape:
            different.append({'key': key, 'shape_mismatch': True})
        elif not torch.equal(a, b):
            different.append({'key': key, 'max_abs': float((a.double()-b.double()).abs().max())})
result = dict(paths=[str(p) for p in paths],keys_match=keys_match,
              tensor_count=len(states[0]),all_tensors_equal=keys_match and not different,
              different_tensors=different)
dest=root/'experiments/structure_ab_100e_20261017/old_new_baseline_weights_audit.json'
dest.write_text(json.dumps(result, indent=2))
print(json.dumps(result))
