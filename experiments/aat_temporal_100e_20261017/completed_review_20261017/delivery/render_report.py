"""Chinese report from the audited machine-readable result, without manual metric transcription."""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE/'comparison.json').read_text(encoding='utf-8'))
NAMES = ['yolov8n', 'yolo11n', 'yolov5nu', 'yolo26n', 'A1_aat_global', 'A2_aat_sequence', 'C1_temporal_uniform', 'C2_temporal_aat_sequence']
LABEL = dict(zip(NAMES, ['YOLOv8n', 'YOLO11n 基线', 'YOLOv5nu', 'YOLO26n', 'A1：全局 AAT', 'A2：序列配额 AAT', 'C1：因果时序输入', 'C2：时序＋序列 AAT']))
K50, K95 = 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
P = lambda x: '—' if x is None else f'{100*x:.2f}'
baseline, c1, c2 = [R['models'][n] for n in ['yolo11n', 'C1_temporal_uniform', 'C2_temporal_aat_sequence']]
lines = []
blocks = []


def title(text, level=2):
    lines.append('#'*level+' '+text+'\n')
    blocks.append(f'<h{level}>{html.escape(text)}</h{level}>')


def paragraph(text):
    lines.append(text+'\n')
    blocks.append('<p>'+html.escape(text)+'</p>')


def table(headers, rows):
    lines.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |')
    lines.extend('| '+' | '.join(map(str,r))+' |' for r in rows)
    lines.append('')
    blocks.append('<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>')


def figure(filename, caption):
    lines.append(f'![{caption}](figures/{filename})\n')
    blocks.append(f'<figure><img src="figures/{html.escape(filename)}"><figcaption>{html.escape(caption)}</figcaption></figure>')


title('CST 基线、A1/A2、C1/C2：完成后统一测试与评价', 1)
paragraph('日期：2026-10-17。下列数值来自实际模型权重和逐帧预测。百分数与“百分点差”分开报告；验证集用于选权重，独立 test 用于最终评价。')
title('先看结论')
paragraph(f'这一轮的有效增量来自时序输入：C1 的 test mAP50 为 {P(c1["test"][K50])}%，相对 YOLO11n 基线为 {c1["test_delta_ap50_pp"]:+.2f} 个百分点；C2 为 {P(c2["test"][K50])}%，相对基线为 {c2["test_delta_ap50_pp"]:+.2f} 个百分点。')
paragraph(f'A1/A2 单独使用 AAT 没有带来点估计收益：相对基线分别为 {R["models"]["A1_aat_global"]["test_delta_ap50_pp"]:+.2f}、{R["models"]["A2_aat_sequence"]["test_delta_ap50_pp"]:+.2f} 个百分点。这两项结果应作为负面消融保留，不能继续把“AAT 提升整体精度”写成已经验证的贡献。')
paragraph('以验证集的精度和误报权衡，建议 C1 作为后续主线，C2 保留作消融：C2 的 val AP50 只比 C1 高约0.56个百分点，val 背景误报却多约3.37个百分点。这个建议依据 val 作出，不按 test 挑模型。')
bt, ct = [r['validation_calibrated_5pct_fpr'] for r in [baseline,c1]]
paragraph(f'还有一个比固定0.25阈值更有工程意义的结果：各模型都仅用 val 选择阈值，使 val 背景误报约为5%，然后原样用于 test。基线的 test 误报为{P(bt["test_background_fpr"])}%、正帧命中为{P(bt["test_positive_hit"])}%；C1 分别为{P(ct["test_background_fpr"])}%和{P(ct["test_positive_hit"])}%。这组观察支持 C1 配合验证集校准具有实用价值；它不抹去预设固定0.25误报门未通过的事实。')
title('统一测试口径')
paragraph('所有模型均使用原生 best.pt，即训练时按 val mAP50–95 选出的权重。没有用 test 选轮次，也没有把 CSV 中 AP50 最大值直接抄进测试表。训练上限100轮、patience=20，提前停止属于既定配置。')
table(['划分', '帧数', '目标数', '背景帧数', '序列数'], [[s,d['frames'],d['objects'],d['negatives'],d['sequences']] for s,d in R['split_counts'].items()])
paragraph('统一推理配置：imgsz=640、矩形补边、batch=128、FP16、conf=0.001、NMS IoU=0.7、max_det=300、workers=2。AP 计算保留全部背景帧。AP50–95 是 IoU 0.50:0.05:0.95，不是 AP50–90。原来的 val 审计用 workers=4，加载线程数不改变该 AP 协议。')
paragraph('实际 test 张量形状已逐批记录并核对：八个模型均为3×544×672。imgsz=640 是配置值，矩形补边后的张量不是640×640。下文复杂度表另按640×640统一估算 GFLOPs，不把它当成本次544×672验证的精确运算量。')
paragraph('模型最佳 F1 处的 P/R 是评测摘要，不能当作统一部署阈值。下表误报率和命中率统一使用固定置信度0.25：误报率=有至少一个预测框的背景帧数÷背景帧数；命中率=至少有一个 IoU≥0.5 正确框的有目标帧数÷有目标帧数。')
title('八个模型的完整对比')
table(['模型','停止轮次','val AP50%','test AP50%','test AP50–95%','test ΔAP50/pp','test 背景误报%','test 正帧命中%'], [[LABEL[n],r['epochs'],P(r['val'][K50]),P(r['test'][K50]),P(r['test'][K95]),f'{r["test_delta_ap50_pp"]:+.2f}',P(r['test_background_fp025']['background_frame_fp_rate']),P(r['test_background_fp025']['positive_frame_hit_rate_iou50'])] for n in NAMES for r in [R['models'][n]]])
figure('01_val_test_ap50.png','同一协议下的 val 与独立 test，保留所有路线；单次训练种子，柱图从0起。')
title('哪些困难属性得到了改善')
attrs = ['TC','SV','DBC','CDB','SIZE_CODE_1_POS_ONLY','SIZE_CODE_2_POS_ONLY','SIZE_CODE_3_POS_ONLY','SIZE_CODE_4_POS_ONLY','OC','OV','FM']
alabel = ['TC 热交叉','SV 尺度变化','DBC 动态背景杂波','CDB 复杂动态背景','SIZE 原始编码1','SIZE 原始编码2','SIZE 原始编码3','SIZE 原始编码4','OC 遮挡','OV 出视野','FM 快速运动']
table(['属性','帧/目标/背景','基线 AP50%','A1','A2','C1','C2'], [[label,f'{base["frames"]}/{base["objects"]}/{base["negative_frames"]}']+[P(R['models'][n]['test_groups'][a].get(K50)) for n in ['yolo11n']+NAMES[4:]] for a,label in zip(attrs,alabel) for base in [baseline['test_groups'][a]]])
paragraph('SV 的含义是尺度变化，不能直接翻译成“小目标”；DBC/CDB 是背景杂波相关属性，不能翻译成明暗差。各属性可以重叠，不能把各组结果当独立样本。SIZE 沿用原始编码，不擅自换成 COCO APsmall。OC/OV 的有效正目标数和 FM 样本量见上表；稀少正目标组的 AP 不用于下强结论。')
figure('02_attribute_differences.png','各路线相对 YOLO11n 的属性 AP50 差值；蓝绿色为正，棕色为负。')
title('精度与误报必须一起看')
paragraph(f'基线 test 背景误报率为 {P(baseline["test_background_fp025"]["background_frame_fp_rate"])}%，C1 为 {P(c1["test_background_fp025"]["background_frame_fp_rate"])}%，C2 为 {P(c2["test_background_fp025"]["background_frame_fp_rate"])}%。因此是否能说“抑制误报”，必须区分固定阈值、校准阈值和具体数据划分，不能仅根据 AP 增长推断。')
paragraph('训练前合同已声明：val 背景误报率最多允许比基线增加2个百分点，标准 recall 最多下降1个百分点。C1/C2 的 val 误报增幅均超过该门槛，因此本轮只能报告精度与误报的权衡，不能写成无代价的全面改善。')
table(['模型','val选定阈值','val 背景误报%','test 背景误报%','test 正帧命中%'], [[LABEL[n],f'{r["threshold"]:.5f}',P(r['val_background_fpr']),P(r['test_background_fpr']),P(r['test_positive_hit'])] for n in ['yolo11n']+NAMES[4:] for r in [R['models'][n]['validation_calibrated_5pct_fpr']]])
paragraph('上表阈值仅从 val 中选择，使 val 背景误报≤5%，再原样应用于 test。test 误报并未被强制为5%，没有在 test 上重新调阈值。这能减小模型置信度标定差异的影响，但仍不是相同 test 误报率下的严格比较。')
figure('03_false_alarms_and_hits.png','左为固定置信度0.25；右为仅在 val 选定阈值后应用于 test。')
title('白盒诊断：模型是否使用了历史信息')
paragraph('C1/C2 不改网络层数，而把原来同一红外图复制出的三通道改为“当前、过去5帧、过去10帧”。只有过去和当前图像，标签仍只属于当前帧，不读取未来帧和历史真值框。test 的历史输入按原规则新生成，与 train/val 的序列严格分离。')
table(['固定权重','正常历史输入 AP50%','历史改成当前帧复制 AP50%','复制减正常/pp'], [[LABEL[n],P(R['models'][n]['test'][K50]),P(d['repeat_current_test_metrics'][K50]),f'{d["repeat_current_minus_normal_ap50_pp"]:+.2f}'] for n,d in R['diagnostics'].items()])
paragraph('如果复制当前帧后变差，说明这些已训练模型依赖历史通道的信息；但这也是一次输入分布改变，不能单凭它证明“运动是唯一原因”，也不能替代从头训练的重复当前帧对照。')
figure('04_history_intervention.png','同一套时序权重，保留正常历史或把三通道全换成当前帧。诊断不是训练消融。')
title('差值对测试序列的敏感性')
table(['比较','观察差值/pp','序列 bootstrap 95%区间/pp'], [[k.replace('__minus__',' − '),f'{v["observed_difference_pp"]:+.2f}',f'[{v["ci95_pp"][0]:+.2f}, {v["ci95_pp"][1]:+.2f}]'] for k,v in R['bootstrap']['contrasts'].items()])
paragraph('以完整序列为单位做配对 bootstrap（60序列、400次），没有把相邻视频帧当独立重复。区间仅反映这组固定权重在 test 序列抽样下的变化；没有覆盖训练随机种子，也不是多比较同时置信区间。本轮未做确认性显著性宣称。')
figure('05_sequence_bootstrap.png','差值和按完整序列重采样得到的描述性区间。')
title('复杂度与当前证据边界')
inventory = json.loads((HERE/'model_inventory.json').read_text(encoding='utf-8'))
table(['模型','未融合参数量/M','640正方形 GFLOPs'], [[LABEL[m['name']],f'{m["stored_parameters"]/1e6:.6f}',f'{m["flops_square640_G"]:.3f}'] for m in inventory])
paragraph('YOLO11n、A1、A2、C1、C2 的未融合参数量均为2,604,883，网络计算图的 GFLOPs 一致。因此可称“网络参数不增加”。时序缓存、输入构造和文件解码仍有成本，不能据此称端到端零延迟。当前为共享 RTX PRO 6000 上的评测，批量耗时不能冒充边缘设备部署延迟。')
paragraph('C1 中途曾收到 SIGKILL，保全第42轮检查点后，从第43轮恢复优化器和 EMA，训练 worker 改为2、验证 worker 改为4，最终第87轮正常早停。其 best.pt 对应第67轮；本轮评测使用的就是这个 best.pt。中断前后随机过程不保证逐位相同。')
paragraph('另一个代码细节：C1 恢复脚本没有继续更新额外的 bestAP50.pt，因此该文件不能代表恢复后的最大 AP50。本报告完全没有使用它，原生 best.pt 正常更新。')
paragraph('基线采用框架的随机加载，A/C 路线采用自定义 EpochSampler；时序路线另用共享亮度增强保持通道语义。当前对比支持“时序输入训练方案整体有效”的工程结论，不能把全部差值唯一归因于某个机制。后续最小补强应是相同加载与亮度设置的重复当前帧训练对照，再补独立种子。')
title('对后续研究的评价')
paragraph('AAT：这轮不适合作为主创新。时序输入：已经有清楚的机制、真实的独立 test 对比和零新增网络参数证据，适合作为后续论文主线。它属于输入设计与训练应用的改进，不能表述成提出了全新的网络结构。是否达到领域 SOTA 或具备充分的新颖性，仍须与同类时序反无人机方法在相同任务和划分下对比。检测 mAP 不能直接与跟踪 mSA 比较。')
paragraph('建议下一步围绕 C1 做小而完整的验证：统一增强/加载的重复帧训练对照、独立种子复验、明确的背景误报控制、batch1端到端计时。C2 的 AAT 贡献先作为待验证的增量，不把两个模块简单叠加就写成已验证协同。')
title('能否投 MVIIPR：定位与最少补强')
paragraph('判断：有值得准备投稿的真实结果，适合走应用与实证研究路线；当前不宜包装成原创时序网络。与用户给出的会议主题匹配，不代表一定录用。最合适的研究问题是：在参数不增加的轻量红外检测器中，因果时序输入是否比困难属性重采样更有效，其精度和误报收益能否迁移到独立视频序列？')
paragraph('文献核查纠正两条旧判断：三帧灰度通道堆叠至少已有 WACVW 2023 的 Exploiting Temporal Context for Tiny Object Detection 和 Sensors 2024 的 Temporal-YOLOv8；而2026-10-06发布的预印本已在 CST 上研究小无人机检测并比较框检测器。因此不能再声称首次提出三帧输入，或首次在 CST 研究检测。只使用过去帧是本实现的在线约束，不作为已经证明的独创方法。')
paragraph('投稿前优先补三件事：①相同 EpochSampler、共享亮度增强、batch和种子设置下，训练“当前帧复制三次”的严格对照，排除训练流程差异；②加入接近已有 Temporal-YOLO 的时序设置或帧差输入对照，并做一组时间间隔消融，明确本应用中的选择依据；③batch1计时，包含缓存与输入构造，报告硬件和测量范围。再做独立种子复验会明显增强说服力。若要主张边缘部署实效，仍需要对应设备的真实测量。')
paragraph('建议标题：因果时序输入在轻量红外反无人机检测中的有效性与误报分析。建议贡献：可复核的轻量检测对比协议；因果输入在该协议下的真实独立test收益；属性、重采样负结果与验证阈值迁移的系统分析。将其写成适配与评测贡献，不列“首次”或“全新网络”。')
refs = [
    ('Corsel et al., WACVW 2023, Exploiting Temporal Context for Tiny Object Detection','https://openaccess.thecvf.com/content/WACV2023W/RWS/html/Corsel_Exploiting_Temporal_Context_for_Tiny_Object_Detection_WACVW_2023_paper.html'),
    ('van Leeuwen et al., Sensors 2024, Toward Versatile Small Object Detection with Temporal-YOLOv8','https://publications.tno.nl/publication/34643397/aHTTXwcv/leeuwen-2024-towards.pdf'),
    ('Xie et al., ICCVW 2025, CST Anti-UAV','https://arxiv.org/abs/2507.23473'),
    ('Son et al., 2026-10-06预印本, What Frame-Level Labels Can and Cannot Do for Small-UAV Point Detection in Thermal Video','https://arxiv.org/abs/2610.07705')]
for text,url in refs:
    lines.append(f'- [{text}]({url})\n')
    blocks.append('<p><a href="'+html.escape(url)+'">'+html.escape(text)+'</a></p>')
title('真实案例：改善与失败一起展示')
for p in sorted((HERE/'figures').glob('case_*.png')):
    figure(p.name,p.stem)
title('复核文件')
paragraph('comparison.json / comparison.csv：全部主表与属性结果；model_inventory.json：权重哈希、配置与复杂度；evaluations/：逐模型审计、逐帧统计和真实预测框；test_data_audit.json：时序 test 输入检查；evaluation_plan.json：评测前固定模型清单；源码脚本保留于同目录。')
paragraph('证据文件的本地哈希、帧数、实际输入形状及11项评测退出状态已核对，见 delivery_integrity.json。所安装技能的 result_card_gate 因缺少共享运行库未能运行，详见 skill_gate_status.json；不将其记作通过，也不将这次描述性结果自动放行成确认性投稿结论。')
(HERE/'CST完成实验_测试对比与评价.md').write_text('\n'.join(lines),encoding='utf-8')
css = 'body{font:16px/1.75 "Microsoft YaHei",sans-serif;max-width:1220px;margin:40px auto;padding:0 24px;color:#17252b;background:#fff}h1{font-size:30px;border-bottom:3px solid #0072b2;padding-bottom:16px}h2{font-size:23px;margin-top:40px}p{max-width:1000px}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #d8e0e5;white-space:nowrap}th{background:#edf3f7}tr:nth-child(even){background:#fafbfc}figure{margin:30px 0}figure img{max-width:100%;height:auto}figcaption{color:#56646d;font-size:13px}a{color:#0072b2}'
(HERE/'CST完成实验_测试对比与评价.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>CST 实验对比与评价</title><style>'+css+'</style><body>'+''.join(blocks)+'</body></html>',encoding='utf-8')
print('REPORT_COMPLETE',HERE/'CST完成实验_测试对比与评价.html')
