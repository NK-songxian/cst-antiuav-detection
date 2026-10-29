# 两个已授权的结构训练任务

已启动并核实真实数据首轮训练和验证：调度器PID 904010，任务A PID 904011、任务B PID 904122。启动检查时A已记录2轮、B已记录1轮，并继续下一轮。对应首轮mAP50分别0.51793、0.51003，仅为起始结果。两路实际优化器均为MuSGD，模型参数分别2,605,459和1,962,531；新增对比支路已产生非零更新。当前状态以远程队列为准，本地evidence是启动快照。

双进程使用expandable_segments分配器缓解显存碎片，实际训练中观察到合计约91GB显存占用；没有更改batch或训练损失。合成显存探针和正式训练证据分别保存。

任务名称固定，不再使用“后两条路线”等模糊说法：

- **A_zero_deploy（任务A）**：原YOLO11n＋可折叠对比卷积。检验部署零增量路线的收益，不预设能涨5点。
- **B_p2_compact（任务B）**：P2/P3/P4预测＋紧凑P2检测头＋可折叠对比卷积。作为mAP50约66.2%→71.2%的主探索方案，5点是目标。

两者原始训练主图25,612张，验证9,066张，100轮上限，patience=20，640，batch256，workers16，seed0，原optimizer=auto、原增强和损失；无AAT。使用项目原始COCO `yolo11n.pt`，没有接续CST已训权重。

远程代码和冻结配置：`/root/autodl-tmp/Ryan/cst_antiuav/experiments/structure_ab_100e_20261017`

输出：`/root/autodl-tmp/Ryan/cst_antiuav/runs/structure_ab_100e_20261017`

队列：输出目录下 `queue/status.json`、`queue/<任务名>.log`。每个任务独立 `results.csv`、`weights/best.pt`、`weights/last.pt`。训练正常结束后导出 `weights/deploy.pt`，检查折叠等价和标准YOLO加载；导出不通过会留下失败证据。

模型创建与权重映射在 `structure_model.py`。A保留标准加载的原始网络，之后添加初始输出为零的训练分支；B显式按语义尺度映射预训练层，P2新层初始化，P3/P4检测塔分别继承旧P3/P4，保留80通道分类中间层，新P2紧凑头为32通道。不能按层号粗暴加载导致尺度错配。具体载入键和参数量保存在各任务 `manifest.json`。

预检查：`preflight.json`记录整模型FP32折叠与保存重载；`memory_*.json`仅是合成输入上的AMP反向和显存测试，不是检测精度。显存探针的反向按batch归一化以避免随机输入的巨大求和梯度；正式训练保持框架原始损失和AMP GradScaler。是否训练正常，以真实数据的训练/验证日志和CSV为准。

加载训练权重best.pt时，Python需能导入本代码目录的自定义模块。导出的deploy.pt移除了自定义类，使用标准DetectionModel及标准PyTorch模块。

只读查看：

```bash
/root/miniconda3/bin/python /root/autodl-tmp/Ryan/cst_antiuav/experiments/structure_ab_100e_20261017/check_status.py
```

队列最多两个训练主进程，启动错开30秒，失败最多重试一次并保留前次输出；退出后只清理本任务进程组。当前入口仅支持本轮新训练；不要直接把resume参数改为true，应先补齐结构恢复与优化器恢复验证。
