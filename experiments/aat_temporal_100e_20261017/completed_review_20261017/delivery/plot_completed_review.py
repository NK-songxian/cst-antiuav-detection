"""Render descriptive, source-linked CST comparisons; no invented error bars."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
R = json.loads((HERE/'comparison.json').read_text(encoding='utf-8'))
OUT = HERE/'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
    'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
NAMES = ['yolov8n', 'yolo11n', 'yolov5nu', 'yolo26n', 'A1_aat_global', 'A2_aat_sequence',
         'C1_temporal_uniform', 'C2_temporal_aat_sequence']
LABELS = ['YOLOv8n', 'YOLO11n', 'YOLOv5nu', 'YOLO26n', 'A1: global AAT',
          'A2: sequence AAT', 'C1: temporal', 'C2: temporal + AAT']
COLORS = ['#999999']*4+['#E69F00', '#CC79A7', '#0072B2', '#009E73']
K50, K95 = 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'


def save(fig, name):
    fig.savefig(OUT/(name+'.png'), dpi=190, facecolor='white', bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'), bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'), bbox_inches='tight')
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.5), layout='constrained')
y = np.arange(len(NAMES))
for ax, split, title in zip(axes, ['val', 'test'], ['Validation (checkpoint selection)', 'Held-out test (final comparison)']):
    scores = [100*R['models'][n][split][K50] for n in NAMES]
    ax.barh(y, scores, color=COLORS, height=.65)
    ax.set(yticks=y, yticklabels=LABELS, xlim=(0, 100), xlabel='mAP50 (%)', title=title)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    for i, score in enumerate(scores):
        ax.text(score+1, i, f'{score:.2f}', va='center', fontsize=9)
fig.suptitle('CST-AntiUAV: fixed validation-selected checkpoints', weight='bold')
fig.supxlabel('One training seed; all negative frames retained; scores are descriptive.', fontsize=9)
save(fig, '01_val_test_ap50')

main = ['yolo11n', 'A1_aat_global', 'A2_aat_sequence', 'C1_temporal_uniform', 'C2_temporal_aat_sequence']
mlabel = ['Baseline', 'A1', 'A2', 'C1', 'C2']
attrs = ['TC', 'SV', 'DBC', 'CDB', 'SIZE_CODE_1_POS_ONLY', 'SIZE_CODE_2_POS_ONLY', 'SIZE_CODE_3_POS_ONLY']
short = ['TC', 'SV', 'DBC', 'CDB', 'Size 1*', 'Size 2*', 'Size 3*']
base = R['models']['yolo11n']['test_groups']
delta = np.asarray([[100*(R['models'][n]['test_groups'][a][K50]-base[a][K50]) for a in attrs] for n in main[1:]])
bound = max(5, np.ceil(np.max(np.abs(delta))))
fig, ax = plt.subplots(figsize=(10.8, 4.4), layout='constrained')
im = ax.imshow(delta, cmap='BrBG', vmin=-bound, vmax=bound, aspect='auto')
for i in range(delta.shape[0]):
    for j in range(delta.shape[1]):
        ax.text(j, i, f'{delta[i,j]:+.2f}', ha='center', va='center',
                color='white' if abs(delta[i,j]) > .65*bound else '#202020', fontsize=11)
ax.set(xticks=np.arange(len(attrs)), xticklabels=short, yticks=np.arange(4), yticklabels=mlabel[1:],
       title='Held-out attribute AP50 difference from YOLO11n (percentage points)')
fig.colorbar(im, ax=ax, label='Difference (pp)', shrink=.8)
counts = ', '.join(f'{s}: {base[a]["objects"]}' for s,a in zip(short, attrs))
fig.supxlabel('Objects per subset: '+counts+'\n*Dataset SIZE codes; overlapping subsets. One training seed, descriptive comparison.', fontsize=8)
save(fig, '02_attribute_differences')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
for n, label, color, marker in zip(main, mlabel, ['#666666']+COLORS[4:], ['o','s','^','D','P']):
    r = R['models'][n]
    fp = r['test_background_fp025']
    axes[0].scatter(100*fp['background_frame_fp_rate'], 100*fp['positive_frame_hit_rate_iou50'], s=110, color=color, marker=marker, label=label)
    t = r['validation_calibrated_5pct_fpr']
    axes[1].scatter(100*t['test_background_fpr'], 100*t['test_positive_hit'], s=110, color=color, marker=marker, label=label)
for ax in axes:
    ax.set(xlim=(0, 30), ylim=(0, 100), xlabel='Background frames with FP (%)', ylabel='Positive frames hit at IoU >= 0.5 (%)')
    ax.grid(alpha=.2)
axes[0].set_title('Held-out test: fixed confidence 0.25')
axes[1].set_title('Held-out test: thresholds calibrated on val')
axes[0].legend(loc='lower right',frameon=False,fontsize=9)
fig.supxlabel('Right: thresholds achieve <=5% val background FP; test FP is unconstrained. Background axis 0–30%; hit axis 0–100%.', fontsize=8)
save(fig, '03_false_alarms_and_hits')

fig, ax = plt.subplots(figsize=(7.8, 4.5), layout='constrained')
x = np.arange(2)
normal = [100*R['models'][n]['test'][K50] for n in main[-2:]]
repeat = [100*R['diagnostics'][n]['repeat_current_test_metrics'][K50] for n in main[-2:]]
ax.bar(x-.18, normal, .36, color='#0072B2', label='Causal history input')
ax.bar(x+.18, repeat, .36, color='#999999', label='Current frame repeated')
for xx, score in zip(np.r_[x-.18, x+.18], normal+repeat):
    ax.text(xx, score+1, f'{score:.2f}', ha='center', fontsize=10)
ax.set(xticks=x, xticklabels=['C1', 'C2'], ylim=(0,100), ylabel='Held-out test mAP50 (%)', title='Does the trained detector use its history channels?')
ax.legend(loc='upper right', frameon=False)
fig.supxlabel('Same fixed weights. Repeated-current input is a distribution-shift diagnostic, not a trained ablation.', fontsize=8)
save(fig, '04_history_intervention')

fig, ax = plt.subplots(figsize=(10, 4.4), layout='constrained')
contrasts = R['bootstrap']['contrasts']
labels = ['A1 - baseline', 'A2 - baseline', 'C1 - baseline', 'C2 - baseline', 'C2 - C1']
for i, (item, label) in enumerate(zip(contrasts.values(), labels)):
    score = item['observed_difference_pp']
    lo, hi = item['ci95_pp']
    ax.plot([lo, hi], [i, i], color='#0072B2', lw=2)
    ax.scatter([score], [i], color='#0072B2', zorder=3)
    ax.text(hi+.25, i, f'{score:+.2f} [{lo:+.2f}, {hi:+.2f}]', va='center', fontsize=9)
ax.axvline(0, color='#777777', ls='--', lw=1)
ax.set(yticks=np.arange(5), yticklabels=labels, xlabel='Test AP50 difference (percentage points)',
       title='Sensitivity to held-out sequence sampling')
ax.invert_yaxis()
ax.margins(x=.35)
fig.supxlabel(f'Paired bootstrap of {R["bootstrap"]["n_sequences"]} sequences, {R["bootstrap"]["replicates"]} replicates. Marginal 95% intervals; one training seed.\nIntervals do not quantify training randomness or establish SOTA.', fontsize=8)
save(fig, '05_sequence_bootstrap')
print('FIGURES_COMPLETE', OUT)
