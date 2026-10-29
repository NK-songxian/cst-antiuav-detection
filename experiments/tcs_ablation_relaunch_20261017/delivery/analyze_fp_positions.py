"""Existing frozen test predictions only; paired background-frame spatial FP audit."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OLD=Path(r'D:\deskFiles\UAV\aat_temporal_20261017\completed_review')
OUT=HERE/'fp_positions';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none'})
shapes=json.loads((HERE/'background_shapes.json').read_text(encoding='utf-8'))
names=['yolo11n','C1_temporal_uniform'];summaries={};histograms={};frameflags={};sources={}
for name in names:
    folder=OLD/'evaluations/test'/name
    pred=json.loads((folder/'predictions_conf010.json').read_text(encoding='utf-8'))
    raw=np.load(folder/'frame_stats.npz',allow_pickle=False)
    stems=raw['frame_stems'];positive=set(raw['target_image_idx'].tolist())
    assert set(shapes)=={str(s) for i,s in enumerate(stems) if i not in positive}
    maximum=np.full(len(stems),-np.inf);np.maximum.at(maximum,raw['pred_image_idx'],raw['conf'])
    raw_flags={str(stems[i]):maximum[i]>=.25 for i in range(len(stems)) if i not in positive}
    frameflags[name]={}
    centers=[];boxes=0
    for stem,(w,h) in shapes.items():
        row=pred[stem];kept=[b for b,c in zip(row['xyxy'],row['conf']) if c>=.25]
        frameflags[name][stem]=bool(kept);boxes+=len(kept)
        assert frameflags[name][stem]==raw_flags[stem],(name,stem)
        centers.extend([[(b[0]+b[2])/2/w,(b[1]+b[3])/2/h] for b in kept])
    pts=np.clip(np.asarray(centers).reshape(-1,2),0,1)
    hist=np.histogram2d(pts[:,1],pts[:,0],bins=(24,30),range=((0,1),(0,1)))[0]
    histograms[name]=hist/len(shapes)*1000
    flagged=sum(frameflags[name].values())
    summaries[name]=dict(background_frames=len(shapes),fp_frames=flagged,fp_frame_rate=flagged/len(shapes),
                          false_boxes=boxes,false_boxes_per_1000_frames=boxes/len(shapes)*1000)
    sources[name]=dict(prediction_sha256=hashlib.sha256((folder/'predictions_conf010.json').read_bytes()).hexdigest(),
                       audit_sha256=hashlib.sha256((folder/'audit.json').read_bytes()).hexdigest())
b=np.array(list(frameflags[names[0]].values()));c=np.array(list(frameflags[names[1]].values()))
paired=dict(removed_fp_frames=int((b&~c).sum()),new_fp_frames=int((~b&c).sum()),
            both_fp=int((b&c).sum()),neither_fp=int((~b&~c).sum()))
fig,axs=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
vmax=max(float(x.max()) for x in histograms.values());diff=histograms[names[0]]-histograms[names[1]]
for name,label,ax in zip(names,['单帧基线','C1：时序通道堆叠'],axs):
    m=ax.imshow(histograms[name],extent=(0,1,1,0),vmin=0,vmax=vmax,cmap='cividis',aspect='auto')
    s=summaries[name]
    ax.set_title(f'{label}\n背景误报帧 {s["fp_frames"]}/{len(shapes)}（{s["fp_frame_rate"]*100:.2f}%）')
    ax.set(xlabel='归一化横坐标',ylabel='归一化纵坐标')
fig.colorbar(m,ax=axs[:2],shrink=.75,label='每1000张背景图的误报框数 / 网格')
lim=max(abs(diff.min()),abs(diff.max()))
m=axs[2].imshow(diff,extent=(0,1,1,0),vmin=-lim,vmax=lim,cmap='PuOr_r',aspect='auto')
axs[2].set_title('基线 − C1\n正值：此区域误报减少');axs[2].set(xlabel='归一化横坐标',ylabel='归一化纵坐标')
fig.colorbar(m,ax=axs[2],shrink=.75,label='误报框数差 / 1000张背景图')
fig.supxlabel('同一2592张test背景帧 · conf≥0.25 · 同一NMS协议 · 30×24网格 · 两模型共用计数色阶',fontsize=9)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'背景误报位置对比.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
labels=['基线有、C1无','两者都有','基线无、C1有','两者都无'];values=[paired[k] for k in ['removed_fp_frames','both_fp','new_fp_frames','neither_fp']]
bars=ax.bar(labels,values,color=['#087F8C','#C7B176','#CC7933','#AEBBC6'])
ax.bar_label(bars,padding=4);ax.set_ylim(0,max(values)*1.14);ax.set_ylabel('背景帧数');ax.set_title('逐帧配对：既报告消除的误报，也报告新增误报')
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'背景误报配对变化.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
case=OLD/'figures/case_2_temporal_removes_background_fp.png'
if case.exists():shutil.copy2(case,OUT/'同帧真实预测示例.png')
result=dict(protocol=dict(split='test',frames=len(shapes),background_definition='original exist=0 annotations',confidence=.25,
    input_coords='original image coordinates normalized separately per frame',spatial_bins=[30,24],
    note='Descriptive analysis of already inspected test results; not new holdout validation or a causal clutter identification experiment'),
    models=summaries,paired=paired,sources=sources,shapes_sha256=hashlib.sha256((HERE/'background_shapes.json').read_bytes()).hexdigest())
(OUT/'fp_position_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
sb=summaries[names[0]];sc=summaries[names[1]]
text=f'''# C1背景误报位置与配对分析

来自已经完成的同一批2592张test背景帧，固定conf≥0.25，复用冻结预测，不重新调阈值。

| 模型 | 有误报背景帧 | 误报帧率 | 误报框总数 |
|---|---:|---:|---:|
| YOLO11n单帧 | {sb['fp_frames']} | {sb['fp_frame_rate']*100:.2f}% | {sb['false_boxes']} |
| C1时序输入 | {sc['fp_frames']} | {sc['fp_frame_rate']*100:.2f}% | {sc['false_boxes']} |

逐帧比较：{paired['removed_fp_frames']}张从有误报变成无误报；{paired['new_fp_frames']}张出现新增误报；{paired['both_fp']}张两者都有误报。

**固定0.25阈值下，C1背景误报帧率增加了{(sc['fp_frame_rate']-sb['fp_frame_rate'])*100:.2f}个百分点，净新增{paired['new_fp_frames']-paired['removed_fp_frames']}张误报帧。不能将主张写为“统一阈值下全面抑制背景”。已有召回/AP收益及按val校准阈值后的工作点需要分开报告。**

![空间计数图](背景误报位置对比.png)

![配对变化](背景误报配对变化.png)

![真实同帧案例](同帧真实预测示例.png)

能支持的结论：在相同阈值与相同背景帧上，C1改变了误报的发生率和空间分布；同时报告新增失败，避免只展示成功。

不能仅凭本图证明：模型识别的是运动、特定云边或建筑纹理。跨场景的归一化图像坐标不是同一物理位置，位置热图不是注意力图。这里不再使用已被证伪的“旧目标残影”解释。完整的运动依赖判断应结合已有repeat-current输入干预和同帧预测对比。

本分析使用已经看过的test结果，为描述性补充；不能写成新方法选择后的独立盲测。
'''
(OUT/'背景误报分析.md').write_text(text,encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
