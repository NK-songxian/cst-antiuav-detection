"""Untrained architecture feasibility only: parameter/FLOP budget and exact folding.

Prior art: P2 scale redistribution; DBB/central-difference reparameterization.
No accuracy claim is made by constructing or profiling these modules.
"""
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv, DWConv
from ultralytics.utils.torch_utils import get_flops

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
OUT = ROOT / 'experiments/structure_plan_20261017'


class FoldableContrastDW(nn.Module):
    """DW 3x3 + learned zero-DC contrast branch; one DW 3x3 after folding."""
    def __init__(self, original):
        super().__init__()
        self.conv = copy.deepcopy(original.conv)
        self.bn = copy.deepcopy(original.bn)
        self.act = copy.deepcopy(original.act)
        c = self.conv.in_channels
        assert self.conv.groups == c == self.conv.out_channels
        assert self.conv.kernel_size == (3, 3) and self.conv.stride == (1, 1)
        self.contrast = nn.Conv2d(c, c, 1, groups=c, bias=False)
        self.contrast_bn = nn.BatchNorm2d(c)
        # Preserve the original function initially; BN scale can learn whether useful.
        nn.init.zeros_(self.contrast_bn.weight)
        nn.init.zeros_(self.contrast_bn.bias)

    def forward(self, x):
        if hasattr(self, 'reparam'):
            return self.act(self.reparam(x))
        detail = x - F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=True)
        return self.act(self.bn(self.conv(x)) + self.contrast_bn(self.contrast(detail)))

    @staticmethod
    def fold(weight, bn):
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return weight * scale[:, None, None, None], bn.bias - bn.running_mean * scale

    def switch_to_deploy(self):
        w0, b0 = self.fold(self.conv.weight, self.bn)
        kernel = self.contrast.weight.new_full((3, 3), -1 / 9)
        kernel[1, 1] += 1
        wd = self.contrast.weight * kernel
        w1, b1 = self.fold(wd, self.contrast_bn)
        c = self.conv.in_channels
        self.reparam = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=True).to(w0)
        with torch.no_grad():
            self.reparam.weight.copy_(w0 + w1)
            self.reparam.bias.copy_(b0 + b1)
        del self.conv, self.bn, self.contrast, self.contrast_bn
        return self


def p2_cfg(original, width):
    cfg = copy.deepcopy(original)
    cfg['head'] = cfg['head'][:6] + [
        [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
        [[-1, 2], 1, 'Concat', [1]],
        [-1, 2, 'C3k2', [width * 4, False]],
        [-1, 1, 'Conv', [256, 3, 2]],
        [[-1, 16], 1, 'Concat', [1]],
        [-1, 2, 'C3k2', [256, False]],
        [-1, 1, 'Conv', [512, 3, 2]],
        [[-1, 13], 1, 'Concat', [1]],
        [-1, 2, 'C3k2', [512, False]],
        [[19, 22, 25], 1, 'Detect', ['nc']]]
    return cfg


def add_contrast(model):
    # Classification only; leave box regression and its loss unchanged.
    head = model.model[-1]
    replacements = []
    for level in [0, 1]:
        branch = head.cv3[level][0]
        original = branch[0]
        branch[0] = FoldableContrastDW(original)
        replacements.append(dict(level=level, channels=original.conv.in_channels,
                                 training_extra_parameters=3 * original.conv.in_channels))
    return replacements


def align_classifier_width(model, width):
    """Actual baseline retains COCO's hidden width=80 despite nc=1; match it."""
    h = model.model[-1]
    channels = [b[0].conv.in_channels for b in h.cv2]
    h.cv3 = nn.ModuleList(nn.Sequential(
        nn.Sequential(DWConv(c, c, 3), Conv(c, width, 1)),
        nn.Sequential(DWConv(width, width, 3), Conv(width, width, 1)),
        nn.Conv2d(width, 1, 1)) for c in channels)


def lighten_p2_regression(model, compact=False):
    h = model.model[-1]
    c = h.cv2[0][0].conv.in_channels
    hidden = 32 if compact else h.cv2[0][0].conv.out_channels
    h.cv2[0] = nn.Sequential(DWConv(c, c, 3), Conv(c, hidden, 1),
                            DWConv(hidden, hidden, 3), Conv(hidden, hidden, 1),
                            nn.Conv2d(hidden, 4 * h.reg_max, 1))
    if compact:
        h.cv3[0] = nn.Sequential(
            nn.Sequential(DWConv(c, c, 3), Conv(c, 32, 1)),
            nn.Sequential(DWConv(32, 32, 3), Conv(32, 32, 1)),
            nn.Conv2d(32, 1, 1))


def folding_checks():
    from ultralytics.nn.modules import DWConv
    torch.manual_seed(42)
    errors = []
    for c, h, w in [(8, 17, 19), (32, 40, 40)]:
        m = FoldableContrastDW(DWConv(c, c, 3)).eval()
        with torch.no_grad():
            m.contrast_bn.weight.uniform_(-1, 1)
            m.contrast_bn.bias.uniform_(-0.1, 0.1)
            m.contrast_bn.running_mean.uniform_(-0.2, 0.2)
            m.contrast_bn.running_var.uniform_(0.3, 1.5)
            m.bn.running_mean.uniform_(-0.2, 0.2)
            m.bn.running_var.uniform_(0.3, 1.5)
        x = torch.randn(2, c, h, w)
        y = m(x)
        m.switch_to_deploy()
        z = m(x)
        error = float((y - z).detach().abs().max())
        assert torch.allclose(y, z, atol=3e-6, rtol=2e-5), error
        errors.append(dict(channels=c, height=h, width=w, max_abs_error=error))
    return errors


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    ref = YOLO(str(ROOT / 'runs/baselines_100e_20261017/yolo11n/weights/best.pt'))
    cfg = copy.deepcopy(ref.model.yaml)
    classifier_width = ref.model.model[-1].cv3[0][0][1].conv.out_channels
    reports = []
    for name, spec, contrast, lite in [
        ('baseline_actual', None, False, False),
        ('baseline_contrast_only', None, True, False),
        ('P2P3P4_width32_standard_head', p2_cfg(cfg, 32), False, False),
        ('P2P3P4_width32_lite_head', p2_cfg(cfg, 32), False, True),
        ('P2P3P4_width32_lite_head_contrast', p2_cfg(cfg, 32), True, True),
        ('P2P3P4_width32_compact_head_contrast', p2_cfg(cfg, 32), True, 'compact')]:
        if spec is None:
            model = copy.deepcopy(ref.model).float()
        else:
            model = DetectionModel(spec, ch=3, nc=1, verbose=False)
            align_classifier_width(model, classifier_width)
        if lite:
            lighten_p2_regression(model, compact=lite == 'compact')
        changed = add_contrast(model) if contrast else []
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            y = model(torch.zeros(1, 3, 640, 640))
        # Profiler supports standard layers; functional pooling/subtraction in the
        # unfused contrast branch is not fully counted. Report the fused graph.
        if contrast:
            for m in list(model.modules()):
                if isinstance(m, FoldableContrastDW):
                    m.switch_to_deploy()
        model.fuse(verbose=False)
        deploy_params = sum(p.numel() for p in model.parameters())
        gflops = float(get_flops(model, imgsz=640))
        assert gflops > 0, 'FLOP profiling unavailable'
        row = dict(name=name, training_parameters=params, fused_parameters=deploy_params,
                   fused_gflops_640_square=gflops, strides=model.stride.tolist(),
                   output_shape=list(y[0].shape), replacements=changed,
                   status='UNTRAINED_FEASIBILITY_ONLY')
        reports.append(row)
        print(json.dumps(row), flush=True)
        del model
    result = dict(status='NO_NEW_TRAINING_OR_ACCURACY_RESULT', reports=reports,
                  classifier_hidden_width=classifier_width,
                  folding_checks=folding_checks(),
                  note='P5 backbone context is retained; P5 prediction is replaced by P2. FLOPs are estimates, not latency measurements.')
    path = OUT / 'feasibility.json'
    if path.exists() and not (OUT / 'feasibility_initial_auto_width.json').exists():
        (OUT / 'feasibility_initial_auto_width.json').write_bytes(path.read_bytes())
    path.write_text(json.dumps(result, indent=2))
    import yaml
    (OUT / 'p2p3p4_width32.yaml').write_text(yaml.safe_dump(p2_cfg(cfg, 32), sort_keys=False))


if __name__ == '__main__':
    main()
