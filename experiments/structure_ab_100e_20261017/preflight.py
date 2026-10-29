"""Block, whole-model folding, serialization, transfer and AMP memory checks."""
import argparse
import copy
import json
import tempfile
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.cfg import get_cfg

from structure_feasibility import FoldableContrastDW, folding_checks
from structure_model import build_from_source

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY = ROOT / 'experiments/structure_ab_100e_20261017'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--memory', choices=['A_zero_deploy', 'B_p2_compact'])
    args = ap.parse_args()
    torch.set_num_threads(2)
    source = YOLO(str(ROOT / 'yolo11n.pt')).model.float()
    if args.memory:
        torch.manual_seed(0)
        model = build_from_source(source, args.memory).cuda().train()
        model.args = get_cfg()
        for p in model.parameters():
            p.requires_grad_(True)
        batch_size = 256
        # Synthetic batch ONLY measures memory/numerics, never detection accuracy.
        x = torch.rand(batch_size, 3, 640, 640, device='cuda')
        batch = {'img': x, 'batch_idx': torch.arange(batch_size, device='cuda').repeat_interleave(4),
                 'cls': torch.zeros(batch_size * 4, 1, device='cuda'),
                 'bboxes': torch.tensor([[.25, .25, .02, .02], [.75, .25, .02, .02],
                                         [.25, .75, .02, .02], [.75, .75, .02, .02]], device='cuda').repeat(batch_size, 1)}
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', dtype=torch.float16):
            loss, items = model(batch)
            scalar = loss.sum()
        # Synthetic random inputs can produce a large summed batch loss. Normalize
        # ONLY this memory probe's backward; full training keeps the framework loss.
        (scalar / batch_size).backward()
        torch.cuda.synchronize()
        assert torch.isfinite(scalar) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        gammas = [m.contrast_bn.weight.grad for m in model.modules() if isinstance(m, FoldableContrastDW)]
        assert any(g is not None and g.abs().sum() > 0 for g in gammas), 'Contrast branch has no gradient'
        result = dict(status='PASS_SYNTHETIC_MEMORY_ONLY', variant=args.memory, batch=256,
                      objects_per_image=4, peak_allocated=torch.cuda.max_memory_allocated(),
                      peak_reserved=torch.cuda.max_memory_reserved(),
                      contrast_gradient_nonzero=True, loss_finite=True,
                      probe_backward_divisor=batch_size, training_loss_unchanged=True)
        (STUDY / ('memory_' + args.memory + '.json')).write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return
    records = []
    blocks = folding_checks()
    for variant, expected in [('A_zero_deploy', 2605459), ('B_p2_compact', 1962531)]:
        torch.manual_seed(0)
        model = build_from_source(source, variant).eval()
        assert sum(p.numel() for p in model.parameters()) == expected
        if variant == 'B_p2_compact':
            assert model.stride.tolist() == [4., 8., 16.]
            assert model.model[-1].cv3[0][-1].in_channels == 32
            assert model.model[-1].cv3[1][-1].in_channels == 80
            # Explicit roles map P3->P3 and P4->P4, rather than layer-index guesses.
            assert any(r.get('destination') == 'head.cv2.1' and r.get('source') == 'head.cv2.0'
                       for r in model.structure_transfer)
        # Use nonzero trained-like branch values to make folding verification meaningful.
        for m in model.modules():
            if isinstance(m, FoldableContrastDW):
                with torch.no_grad():
                    m.contrast_bn.weight.uniform_(-.2, .2)
        x = torch.rand(1, 3, 160, 160)
        with torch.no_grad():
            y = model(x)[0]
        with tempfile.TemporaryDirectory(dir=STUDY) as tmp:
            p = Path(tmp) / 'roundtrip.pt'
            torch.save({'model': model}, p)
            restored = torch.load(p, weights_only=False)['model'].eval()
            with torch.no_grad():
                z = restored(x)[0]
            torch.testing.assert_close(y, z, atol=0, rtol=0)
        fused = copy.deepcopy(model).fuse(verbose=False).eval()
        with torch.no_grad():
            z = fused(x)[0]
        torch.testing.assert_close(y, z, atol=1e-3, rtol=1e-4)
        error = float((y-z).abs().max())
        records.append(dict(variant=variant, params=expected, strides=model.stride.tolist(),
                            serialization_exact=True, whole_model_folding_max_abs_error=error,
                            fold_pass=True))
    (STUDY / 'preflight.json').write_text(json.dumps(dict(status='PASS', models=records, blocks=blocks), indent=2))
    print(json.dumps(records), flush=True)


if __name__ == '__main__':
    main()
