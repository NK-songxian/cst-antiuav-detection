# CST Anti-UAV检测评测协议与三帧时序输入

AutoDL项目`/root/autodl-tmp/Ryan/cst_antiuav`的代码与实验记录快照，导出日期：2026-10-29。

本仓库保留数据格式转换、原序列划分清单、单帧/三帧输入、训练、评测、统计与绘图代码，以及非权重实验配置和结果。数据集、训练权重与预训练权重需另行准备。

## 主要入口

| 工作 | 文件 |
|---|---|
| CST跟踪标注转YOLO框标签、保留空标签与属性索引 | `scripts/prep_cst_yolo.py` |
| 原训练/验证/测试序列及抽样样本清单 | `splits/` |
| 从属性索引重新导出划分清单 | `tools/export_splits.py` |
| 三灰度通道构建与早期对照 | `experiments/aat_temporal_100e_20261017/prepare_study.py` |
| 帧间隔输入构建 | `experiments/tcs_ablation_20261017/build_data.py` |
| 统一图像组单帧/三帧运行逻辑 | `experiments/cst_paper_completion_20261018/matched_runtime.py` |
| 配对输入训练与任务队列 | 同目录的`train_matched.py`、`queue_matched.py` |
| 检测、背景误报及重新出现评测 | 同目录的`evaluate_matched.py` |
| 帧间隔验证与bootstrap | `experiments/cst_gap_confirmation_20261019/` |
| TC加权采样 | `experiments/aat_temporal_100e_20261017/sampling.py` |
| 数据与验证流程审计 | `experiments/baselines_100e_20261017/audit_validation.py` |
| 论文绘图 | `paper_assets_figures_20261027/` |
| 上游CST跟踪评测工具 | `code/CST-Anti-UAV/` |

## 数据划分

沿用CST原有序列划分，不进行帧级随机重划分。`splits/`从本次实际使用的`data/yolo/attribute_index.json`提取，仅包含序列名、抽样样本名与汇总计数，不包含原始图像。

| 划分 | 序列数 | 抽样样本数 | 有目标 | 无目标 |
|---|---:|---:|---:|---:|
| train | 120 | 25,612 | 22,000 | 3,612 |
| val | 40 | 9,066 | 7,819 | 1,247 |
| test | 60 | 14,446 | 11,854 | 2,592 |

抽样步长为5个原视频帧。转换器在`exist=1`且框宽、高均大于1像素时输出YOLO框，否则输出空标签；原存在状态和属性另存索引。该转换器按原实验的640×512图像尺寸归一化。索引中的`frame`为转换器使用的零基帧索引。

## 准备与使用

1. 准备CST原始目录，使其包含`train/`、`val/`、`test/`及序列内的JPEG和`IR_label.json`。
2. 参考`environment_snapshot.json`、`requirements_snapshot.txt`与各实验`contract.json`安装环境。当前导出环境中的Ultralytics为8.4.177；PyTorch/CUDA安装需匹配实际机器。早期`scripts/setup_env.sh`保留原样，原脚本中的版本与当前环境并不完全相同。
3. 转换数据：

```bash
python scripts/prep_cst_yolo.py   --cst_root /path/to/CST-AntiUAV   --out "$PWD/data/yolo"   --frame_stride 5

python tools/export_splits.py   --index data/yolo/attribute_index.json   --out split_check
```

4. 核对`split_check`与仓库`splits/`，再由对应的`prepare_study.py`或`build_data.py`准备时序输入。
5. 将所需预训练权重放回原配置指定的位置。原实验脚本保留AutoDL绝对路径；迁移机器时需调整各脚本的`ROOT`、数据配置、`common_args.json`和`matrix.json`路径。
6. 在新的空输出目录运行训练。`runs/`和`experiments/`保留历史结果；不要用原任务名覆盖这些记录。部分`prepare_*`脚本是当时运行任务的编排快照，含进程状态断言，不应直接当成新环境的一键启动器。

例如，完成路径、权重和输出目录配置后，统一图像组训练入口为：

```bash
python experiments/cst_paper_completion_20261018/train_matched.py   --study experiments/cst_paper_completion_20261018   --name N_current_seed0
```

单帧复制当前灰度图至三个通道；主要三帧设置使用当前及两幅历史图像，原帧偏移为0/5/10，仅监督当前帧。原流程组与统一图像组分别保留配置和结果，输入增益在统一图像组内比较。

## 打包范围与来源

保留原项目目录层次。排除根`data/`、数据图像/标签目录、数据软链接、91个模型权重文件、旧压缩包和运行缓存。保留数据划分清单、配置、指标、采样记录、分析结果与绘图材料；具体文件及哈希见`EXPORT_MANIFEST.json`。本次只检查代码语法和文件完整性，没有重新运行训练。

上游CST代码与许可证原样位于`code/CST-Anti-UAV/`。本快照没有替作者另行授予整个仓库的新许可证；使用第三方代码时遵循对应目录的原始许可。
