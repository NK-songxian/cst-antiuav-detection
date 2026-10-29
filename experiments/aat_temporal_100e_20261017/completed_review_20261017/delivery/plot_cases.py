"""Draw actual predictions alongside current and historical frames."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE/'qualitative/cases.json').read_text(encoding='utf-8'))
OUT = HERE/'figures'
NAMES = ['yolo11n', 'C1_temporal_uniform', 'C2_temporal_aat_sequence']
LABELS = ['Single-frame baseline', 'C1: temporal', 'C2: temporal + AAT']
TITLES = {'temporal_recovers_target':'Target recovered by temporal models',
          'temporal_removes_background_fp':'A baseline background false alarm removed',
          'remaining_temporal_background_fp':'Background false alarm remains / appears',
          'temporal_regression':'A regression: temporal model misses a baseline hit'}
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})

for idx, case in enumerate(DATA['cases']):
    if case['selected'] is None:
        continue
    image = Image.open(HERE/'qualitative/images'/(case['selected']+'.jpg'))
    width, height = image.size
    gt = []
    for line in case['yolo_ground_truth'].splitlines():
        _, x, y, w, h = map(float,line.split())
        gt.append(((x-w/2)*width,(y-h/2)*height,w*width,h*height))
    fig, axes = plt.subplots(2,3,figsize=(12,7.6),layout='constrained')
    for j, (stem, ax) in enumerate(zip(case['history'],axes[0])):
        ax.imshow(Image.open(HERE/'qualitative/images'/(stem+'.jpg')),cmap='gray')
        ax.set_title(['History: t-10','History: t-5','Current: t'][j])
        ax.axis('off')
    for name, label, ax in zip(NAMES, LABELS, axes[1]):
        ax.imshow(image,cmap='gray')
        for x,y,w,h in gt:
            ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#00FFFF',lw=1.6))
        p = case['models'][name]['predictions']
        for box, conf in zip(p['xyxy'],p['conf']):
            if conf < .25:
                continue
            x1,y1,x2,y2 = box
            ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='#FF9C26',lw=1.3))
            ax.text(x1,max(12,y1-4),f'{conf:.2f}',color='#FF9C26',fontsize=8,
                    bbox=dict(facecolor='black',alpha=.65,edgecolor='none',pad=1))
        ax.set_title(label)
        ax.axis('off')
    fig.suptitle(TITLES[case['kind']],weight='bold',fontsize=13)
    fig.supxlabel(case['selected']+' | Cyan: ground truth; orange: predictions at confidence >= 0.25.\nIllustrative case selected after results; frame counts are reported separately.',fontsize=8)
    for ext in ['png','pdf']:
        fig.savefig(OUT/(f'case_{idx+1}_{case["kind"]}.'+ext),dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)
print('QUALITATIVE_FIGURES_COMPLETE')
