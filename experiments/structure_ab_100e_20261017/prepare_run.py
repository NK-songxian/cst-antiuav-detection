"""Freeze two authorized tasks, common recipe, source weights and code hashes."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path('/root/autodl-tmp/Ryan/cst_antiuav')
STUDY = ROOT / 'experiments/structure_ab_100e_20261017'
BASE = ROOT / 'experiments/baselines_100e_20261017'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert json.loads((STUDY / 'preflight.json').read_text())['status'] == 'PASS'
    memory = [json.loads((STUDY / ('memory_' + name + '.json')).read_text())
              for name in ['A_zero_deploy', 'B_p2_compact']]
    assert all(m['status'] == 'PASS_SYNTHETIC_MEMORY_ONLY' for m in memory)
    peak_sum = sum(m['peak_reserved'] for m in memory)
    total = torch.cuda.get_device_properties(0).total_memory
    safety = 2 * 1024**3  # CUDA contexts live outside PyTorch's reserved allocator.
    assert peak_sum + safety < total, f'Two-job peaks plus context reserve exceed GPU: {peak_sum}, {total}'
    original = json.loads((BASE / 'contract.json').read_text())
    common = json.loads((BASE / 'common_args.json').read_text())
    common['project'] = str(ROOT / 'runs/structure_ab_100e_20261017')
    (STUDY / 'common_args.json').write_text(json.dumps(common, indent=2))
    weights = ROOT / 'yolo11n.pt'
    models = [dict(name=name, weights=str(weights), sha256=sha(weights))
              for name in ['A_zero_deploy', 'B_p2_compact']]
    contract = dict(models=models, baseline_study=str(BASE), splits=original['splits'],
                    environment=original['environment'], runtime=original['runtime'],
                    max_parallel=2, task_A='Original YOLO11n + foldable contrast classification convolution',
                    task_B='P2/P3/P4 compact head + foldable contrast classification convolution',
                    target='B: exploratory +5 percentage points AP50, approximately 71.21%; not a promised result',
                    stopping='100 maximum epochs, patience 20, same AP50-95 checkpoint selection as baseline',
                    common_args_sha256=sha(STUDY / 'common_args.json'),
                    code_sha256={p.name: sha(p) for p in STUDY.glob('*.py')},
                    preflight_sha256=sha(STUDY / 'preflight.json'),
                    synthetic_memory_peak_reserved_sum=peak_sum,
                    gpu_total_memory=total, cuda_context_safety_reserve=safety,
                    data='Original CST manifest only. No AAT, no custom loss, no added data augmentation.',
                    source_authorization='User: OK 并行执行 (2026-10-17)')
    (STUDY / 'contract.json').write_text(json.dumps(contract, indent=2))
    print('PREPARED', 'memory_peak_sum_GiB', peak_sum/1024**3, 'tasks', [m['name'] for m in models])


if __name__ == '__main__':
    main()
