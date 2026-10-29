# -*- coding: utf-8 -*-
"""
Fig1 redraw (English version): single-frame / three-frame input construction +
unified YOLO detection network schematic. Only the text language differs from
the Chinese version; design and values are unchanged. Photo crops come from the
original docx fig1 (fig1_src/); content matches the paper case (YOLO11n, red
boxes = ground truth, orange boxes = predictions).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
SRC = OUT / "fig1_src"

BLUE = "#2E5FA3"
ORANGE = "#E4622F"
GRAY = "#767676"
DARK = "#222222"

# ---- font (English version: DejaVu Sans, no CJK needed) ----------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
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

# ---- lane background ---------------------------------------------------------
for y0, h, c in [(0.575, 0.375, BLUE), (0.035, 0.375, ORANGE)]:
    fig.add_artist(FancyBboxPatch((0.004, y0), 0.992, h,
                   boxstyle="round,pad=0.004,rounding_size=0.012",
                   transform=fig.transFigure, fc=c, ec="none", alpha=0.055, zorder=0))

# ---- lane titles (wrapped to two lines: longer English text) ------------------
fig.text(0.012, 0.755, "Single-frame\ninput", fontsize=9, fontweight="bold",
         color=BLUE, va="center", linespacing=1.15)
fig.text(0.012, 0.215, "Three-frame\ninput", fontsize=9, fontweight="bold",
         color=ORANGE, va="center", linespacing=1.15)

# ---- lane 1: single-frame input (top) -----------------------------------------
PW, PH = 0.075, 0.1125            # photo size (ratio 1.5, 512x640 case image)
top_y = 0.70 - PH / 2             # top-lane photo center y=0.70
img_axes(fig, SRC / "top_t.png", 0.215, top_y, PW, PH)
fig.text(0.2525, top_y - 0.062, "Current frame t", ha="center", fontsize=8, color=DARK)
fig.text(0.2525, top_y - 0.100, "Replicated to 3 channels", ha="center", fontsize=7, color=GRAY)
fig.text(0.2525, top_y + PH + 0.022, "1 × 512 × 640", ha="center", fontsize=6.8, color=GRAY)

# ---- lane 2: three-frame input (bottom) ----------------------------------------
bot_y = 0.215 - PH / 2            # bottom-lane photo center y=0.215
xs = [0.115, 0.212, 0.309]
caps = [("t−10 (10 back)", "Channel 3"), ("t−5 (5 back)", "Channel 2"),
        ("Current frame t", "Channel 1")]
files = ["t_10.png", "t_5.png", "t_now.png"]
for x0, (c1, c2), f in zip(xs, caps, files):
    img_axes(fig, SRC / f, x0, bot_y, PW, PH)
    fig.text(x0 + PW / 2, bot_y - 0.034, c1, ha="center", fontsize=6.5, color=DARK)
    fig.text(x0 + PW / 2, bot_y - 0.074, c2, ha="center", fontsize=7, color=GRAY)
fig.text(0.2495, bot_y + PH + 0.012, "Each frame 1 × 512 × 640", ha="center",
         fontsize=6.8, color=GRAY)

# ---- middle: unified YOLO module ------------------------------------------------
MX0, MX1, MY0, MY1 = 0.435, 0.760, 0.045, 0.925
fig.add_artist(FancyBboxPatch((MX0, MY0), MX1 - MX0, MY1 - MY0,
               boxstyle="round,pad=0.006,rounding_size=0.02",
               transform=fig.transFigure, fc="white", ec="#8A8A8A", lw=1.1, zorder=2))
cx = (MX0 + MX1) / 2
fig.text(cx, 0.845, "Unified lightweight\nYOLO detection network", ha="center",
         va="center", fontsize=8.5, fontweight="bold", color=DARK,
         linespacing=1.2, zorder=3)
fig.text(cx, 0.765, "YOLO11n / YOLO11s · identical configuration", ha="center",
         fontsize=6.6, color=GRAY, zorder=3)

stages = [("Backbone", "Conv. feature\nextraction,\ndownsample"),
          ("Neck", "Multi-scale\nfeature fusion"),
          ("Head", "Class & box\nprediction")]
sw, sh, sy = 0.085, 0.115, 0.545   # stage box width/height/bottom
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
fig.text(cx, 0.315, "Output N × 6 (class logits + box regression)", ha="center",
         fontsize=6.4, color=GRAY, zorder=3)

fig.add_artist(plt.Line2D([MX0 + 0.02, MX1 - 0.02], [0.295, 0.295],
              transform=fig.transFigure, ls=(0, (3, 2)), lw=0.7, color="#B0B0B0", zorder=3))
fig.text(cx, 0.253, "Architecture and parameter count\nunchanged across input schemes",
         ha="center", va="center", fontsize=6.0, color=DARK, linespacing=1.3, zorder=3)
fig.text(cx, 0.170, "Two inputs trained separately\nwith independent weights",
         ha="center", va="center", fontsize=6.0, color=DARK, linespacing=1.3, zorder=3)
fig.text(cx, 0.088, "Temporal scheme only replaces the\ninput tensor: zero extra params / FLOPs",
         ha="center", va="center", fontsize=6.0, color=ORANGE, linespacing=1.3, zorder=3)

# ---- input -> network arrows and labels ------------------------------------------
arrow(fig, 0.295, 0.70, MX0 - 0.004, 0.70, color=BLUE)
fig.text((0.295 + MX0) / 2, 0.735, "3 × 544 × 672", ha="center", fontsize=6.8, color=BLUE)
arrow(fig, 0.389, 0.215, MX0 - 0.004, 0.215, color=ORANGE)
fig.text(0.398, 0.300, "Concatenated\n3 × 544 × 672", ha="center", fontsize=6.5,
         color=ORANGE, linespacing=1.3)

# ---- network -> result arrows and result images -----------------------------------
RX, RW = 0.845, 0.105             # result image width (ratio 1.48)
RH = RW / (169 / 114)
arrow(fig, MX1 + 0.004, 0.70, RX - 0.004, 0.70, color=BLUE)
arrow(fig, MX1 + 0.004, 0.215, RX - 0.004, 0.215, color=ORANGE)
fig.text((MX1 + RX) / 2, 0.735, "N × 6", ha="center", fontsize=6.8, color=BLUE)

img_axes(fig, SRC / "res_single.png", RX, 0.70 - RH / 2, RW, RH)
fig.text(RX + RW / 2, 0.70 - RH / 2 - 0.030, "Not detected", ha="center",
         fontsize=8, color=DARK)
img_axes(fig, SRC / "res_tcs.png", RX, 0.215 - RH / 2, RW, RH)
fig.text(RX + RW / 2, 0.215 - RH / 2 - 0.030, "Correctly detected", ha="center",
         fontsize=8, color=DARK)

# ---- footnote ---------------------------------------------------------------------
fig.text(0.5, 0.008, "Case study with YOLO11n; sizes as C × H × W; red boxes are "
         "ground truth (display only), orange boxes are model predictions.",
         ha="center", fontsize=6.8, color=GRAY)

fig.savefig(OUT / "fig1_input_pipeline_en.png", dpi=300)
print("saved", OUT / "fig1_input_pipeline_en.png")
