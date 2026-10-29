# -*- coding: utf-8 -*-
"""
图1 重绘：单帧 / 三帧输入构建 + 统一 YOLO 检测网络示意图。
照片素材自原 docx 图1 裁剪（fig1_src/），内容与论文案例一致（YOLO11n，红框真值、橙框预测）。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
SRC = OUT / "fig1_src"

BLUE = "#2E5FA3"
ORANGE = "#E4622F"
GRAY = "#767676"
DARK = "#222222"

# ---- 字体（与指标图脚本一致） ------------------------------------------------
cjk_candidates = ["Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC"]
installed = {f.name for f in font_manager.fontManager.ttflist}
if not any(c in installed for c in cjk_candidates):
    for p in sorted(Path("/root/.local/share/fonts").glob("NotoSansCJK*.otf")):
        font_manager.fontManager.addfont(str(p))
    installed = {f.name for f in font_manager.fontManager.ttflist}
chosen = [c for c in cjk_candidates if c in installed][0]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [chosen, "DejaVu Sans"],
    "axes.unicode_minus": False,
})


def img_axes(fig, path, x, y, w, h):
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(plt.imread(str(path)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.7); s.set_color("#BBBBBB")
    return ax


def arrow(fig, x0, y0, x1, y1, color="#555555", lw=1.2):
    a = FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure,
                        arrowstyle="-|>", mutation_scale=9, lw=lw,
                        color=color, shrinkA=0, shrinkB=0, zorder=4)
    fig.add_artist(a)
    return a


fig = plt.figure(figsize=(6.7, 3.0), dpi=300)

# ---- 泳道底色 ----------------------------------------------------------------
for y0, h, c in [(0.575, 0.375, BLUE), (0.035, 0.375, ORANGE)]:
    fig.add_artist(FancyBboxPatch((0.004, y0), 0.992, h,
                   boxstyle="round,pad=0.004,rounding_size=0.012",
                   transform=fig.transFigure, fc=c, ec="none", alpha=0.055, zorder=0))

# ---- 泳道标题 ----------------------------------------------------------------
fig.text(0.012, 0.755, "单帧输入", fontsize=10, fontweight="bold", color=BLUE, va="center")
fig.text(0.012, 0.215, "三帧输入", fontsize=10, fontweight="bold", color=ORANGE, va="center")

# ---- 泳道 1：单帧输入（顶部） -------------------------------------------------
PW, PH = 0.075, 0.1125            # 照片宽高（比例 1.5，512x640 案例图）
top_y = 0.70 - PH / 2             # 顶泳道照片中心 y=0.70
img_axes(fig, SRC / "top_t.png", 0.215, top_y, PW, PH)
fig.text(0.2525, top_y - 0.062, "当前帧 t", ha="center", fontsize=8, color=DARK)
fig.text(0.2525, top_y - 0.088, "复制为 3 通道", ha="center", fontsize=7, color=GRAY)
fig.text(0.2525, top_y + PH + 0.022, "1 × 512 × 640", ha="center", fontsize=6.8, color=GRAY)

# ---- 泳道 2：三帧输入（底部） -------------------------------------------------
bot_y = 0.215 - PH / 2            # 底泳道照片中心 y=0.215
xs = [0.115, 0.212, 0.309]
caps = [("前10帧  t−10", "通道 3"), ("前5帧  t−5", "通道 2"), ("当前帧  t", "通道 1")]
files = ["t_10.png", "t_5.png", "t_now.png"]
for x0, (c1, c2), f in zip(xs, caps, files):
    img_axes(fig, SRC / f, x0, bot_y, PW, PH)
    fig.text(x0 + PW / 2, bot_y - 0.034, c1, ha="center", fontsize=7.5, color=DARK)
    fig.text(x0 + PW / 2, bot_y - 0.056, c2, ha="center", fontsize=7, color=GRAY)
fig.text(0.2495, bot_y + PH + 0.012, "各帧 1 × 512 × 640", ha="center",
         fontsize=6.8, color=GRAY)

# ---- 中部：统一 YOLO 模块 ------------------------------------------------------
MX0, MX1, MY0, MY1 = 0.435, 0.760, 0.045, 0.925
fig.add_artist(FancyBboxPatch((MX0, MY0), MX1 - MX0, MY1 - MY0,
               boxstyle="round,pad=0.006,rounding_size=0.02",
               transform=fig.transFigure, fc="white", ec="#8A8A8A", lw=1.1, zorder=2))
cx = (MX0 + MX1) / 2
fig.text(cx, 0.845, "统一的轻量 YOLO 检测网络", ha="center",
         fontsize=9.5, fontweight="bold", color=DARK, zorder=3)
fig.text(cx, 0.800, "YOLO11n / YOLO11s · 配置一致", ha="center",
         fontsize=7.5, color=GRAY, zorder=3)

stages = [("Backbone", "卷积特征提取\n逐级下采样"),
          ("Neck", "多尺度特征\n融合"),
          ("Head", "类别与位置\n预测")]
sw, sh, sy = 0.085, 0.115, 0.545   # 阶段框宽/高/底边
sx = [0.452, 0.560, 0.668]
for (name, sub), x0 in zip(stages, sx):
    fig.add_artist(FancyBboxPatch((x0, sy), sw, sh,
                   boxstyle="round,pad=0.004,rounding_size=0.012",
                   transform=fig.transFigure, fc="#F2F5FA", ec=BLUE, lw=0.9, zorder=3))
    fig.text(x0 + sw / 2, sy + sh / 2, name, ha="center", va="center",
             fontsize=8.5, color=DARK, zorder=4)
    fig.text(x0 + sw / 2, sy - 0.045, sub, ha="center", va="top",
             fontsize=6.6, color=GRAY, linespacing=1.25, zorder=4)
for x0 in sx[:-1]:
    arrow(fig, x0 + sw + 0.006, sy + sh / 2, x0 + sw + 0.017, sy + sh / 2, color=BLUE, lw=1.0)
fig.text(cx, 0.315, "输出 N × 6（类别置信度 + 框回归）", ha="center",
         fontsize=6.8, color=GRAY, zorder=3)

fig.add_artist(plt.Line2D([MX0 + 0.02, MX1 - 0.02], [0.245, 0.245],
             transform=fig.transFigure, ls=(0, (3, 2)), lw=0.7, color="#B0B0B0", zorder=3))
fig.text(cx, 0.205, "网络结构与参数量不随输入方案改变", ha="center",
         fontsize=7.3, color=DARK, zorder=3)
fig.text(cx, 0.160, "两种输入分别训练，权重相互独立", ha="center",
         fontsize=7.3, color=DARK, zorder=3)
fig.text(cx, 0.105, "时序方案仅替换输入张量，零额外参数 / 零额外 FLOPs", ha="center",
         fontsize=7.3, color=ORANGE, zorder=3)

# ---- 输入 → 网络 箭头与标签 ----------------------------------------------------
arrow(fig, 0.295, 0.70, MX0 - 0.004, 0.70, color=BLUE)
fig.text((0.295 + MX0) / 2, 0.735, "3 × 544 × 672", ha="center", fontsize=6.8, color=BLUE)
arrow(fig, 0.389, 0.215, MX0 - 0.004, 0.215, color=ORANGE)
fig.text(0.412, 0.272, "拼接\n3 × 544 × 672", ha="center", fontsize=6.5,
         color=ORANGE, linespacing=1.3)

# ---- 网络 → 结果 箭头与结果图 --------------------------------------------------
RX, RW = 0.845, 0.105             # 结果图宽（比例 1.48）
RH = RW / (169 / 114)
arrow(fig, MX1 + 0.004, 0.70, RX - 0.004, 0.70, color=BLUE)
arrow(fig, MX1 + 0.004, 0.215, RX - 0.004, 0.215, color=ORANGE)
fig.text((MX1 + RX) / 2, 0.735, "N × 6", ha="center", fontsize=6.8, color=BLUE)

img_axes(fig, SRC / "res_single.png", RX, 0.70 - RH / 2, RW, RH)
fig.text(RX + RW / 2, 0.70 - RH / 2 - 0.030, "未检出目标", ha="center",
         fontsize=8, color=DARK)
img_axes(fig, SRC / "res_tcs.png", RX, 0.215 - RH / 2, RW, RH)
fig.text(RX + RW / 2, 0.215 - RH / 2 - 0.030, "正确检出", ha="center",
         fontsize=8, color=DARK)

# ---- 底注 ----------------------------------------------------------------------
fig.text(0.5, 0.008, "检测案例为 YOLO11n；尺寸按 C × H × W 表示；红框为真值（仅作展示），橙框为模型预测。",
         ha="center", fontsize=6.8, color=GRAY)

fig.savefig(OUT / "fig1_input_pipeline.png", dpi=300)
print("saved", OUT / "fig1_input_pipeline.png")
