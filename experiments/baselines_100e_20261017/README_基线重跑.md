# CST 四模型基线重跑

2026-10-17，按用户要求恢复原始基线训练，四个小模型、两个训练进程并行、任务队列自动补位。

## 模型与统一配置

顺序：YOLOv8n、YOLO11n、YOLOv5nu、YOLO26n。使用各自现有或官网下载的预训练权重；具体 SHA256 记录在远程 contract.json。YOLOv5nu 是 Ultralytics 的 YOLOv5u nano，论文中应写清版本，不能与原版 YOLOv5n 混称。

最多 100 epochs，patience=20，imgsz=640，batch=256，workers=16，seed=0，AMP 开启。优化器保持原配置 optimizer=auto，实际选出的优化器记录在各任务 manifest.json。其余框架可识别的训练配置继承原 YOLO11n 的 args.yaml；参考原文件单独保留。

数据恢复原始训练清单：训练 25,612 帧（含 3,612 背景帧），验证 9,066 帧（含 1,247 背景帧），测试 14,446 帧。训练期间仅在验证集评估。清单排除 AAT 复制链接，未加入 FP 配对损失、蒸馏或额外增强。原生 Mosaic 等增强按原配置保留。

为限制双进程内存和清理残留子进程，DataLoader 使用 prefetch_factor=1、pin_memory=False、forkserver；这些是运行时调整，已记录，不能称整个运行环境逐字节相同。

## 远程路径

- 项目：`/root/autodl-tmp/Ryan/cst_antiuav`
- 代码及配置：`/root/autodl-tmp/Ryan/cst_antiuav/experiments/baselines_100e_20261017`
- 新输出：`/root/autodl-tmp/Ryan/cst_antiuav/runs/baselines_100e_20261017`
- 队列状态：新输出目录下 `queue/status.json`
- 训练日志：新输出目录下 `queue/<模型名>.log`
- 每轮指标：新输出目录下 `<模型名>/results.csv`
- 调度器初始 PID：840648；后续以 `queue/status.json` 为准。

## 清理与恢复

已清理 39 个经进程命令和目录确认属于本项目的残留进程，清理后曾核实 GPU 占用 0 MiB。旧 runs 整体移到以下归档目录，未删除 FP pool、旧权重和实验记录：

`/root/autodl-tmp/Ryan/cst_antiuav/archive/baseline_reset_20261017_003756/runs`

清理明细和原输出位置在项目 `baseline_reset_latest.json` 及归档 `cleanup_processes.json`。

## 队列行为

同时最多两个训练主进程，启动间隔 30 秒。任务正常训练返回、退出码为零、完成记录与权重和有效 CSV 同时存在，才记为完成。每个失败任务最多重试一次，失败输出保留在独立目录。主进程退出后仅清理该任务所属进程组，避免 DataLoader 子进程继续占显存。完成判断的六个针对性检查已通过。

只读查看状态：

```bash
/root/miniconda3/bin/python /root/autodl-tmp/Ryan/cst_antiuav/experiments/baselines_100e_20261017/check_status.py
```

`launch.json`、`contract.json`、`common_args.json`、各任务 `manifest.json` 与 `results.csv` 为核对依据。启动不等于训练完成；本说明不预设最终指标。

## 已核实的启动结果

北京时间 2026-10-17 00:48 左右，YOLOv8n 已记录 2 轮、YOLO11n 已记录 1 轮，两路均继续训练。两者第一轮验证集 mAP50 分别为 0.50867、0.53808，mAP50-95 分别为 0.24205、0.24690，仅代表首轮，不能当作最终基线。GPU 占用约 71,878 MiB，利用率快照 98%；此数字是训练占用，不能当作推理延迟或部署显存。其余两个模型等待队列补位。启动快照及当时 CSV 已下载到本地 evidence 目录，后续状态以远程文件为准。
