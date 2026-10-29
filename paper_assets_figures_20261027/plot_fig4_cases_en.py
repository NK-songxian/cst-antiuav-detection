# English rebuild of the response-map case-study figure (fig 4 in the CST
# manuscript). Source: /tmp/docx_media/image4.png (2x3 panels + power-law
# colorbar). Panels were segmented into fig4_src/ (3 px border trimmed). The
# burned-in Chinese score captions are covered and re-rendered in English; the
# colorbar is rebuilt natively from colors sampled off the source bar, with tick
# fractions measured from the source tick labels.
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from PIL import Image

OUT = Path(__file__).resolve().parent
SRC = OUT / "fig4_src"
SRC_IMG = Path("/tmp/docx_media/image4.png")

DARK = "#222222"

# ---- geometry measured on the source image (1472 x 632 px) --------------------
SW, SH = 1472.0, 632.0
W_IN = 17 / 2.54                 # 17 cm target width
FIGSIZE = (W_IN, W_IN * SH / SW)

# panel content bboxes (border trimmed), top-down px: (x0, x1, y0, y1)
PANELS = {
    (0, 0): (153, 531, 46, 295), (0, 1): (553, 931, 46, 295),
    (0, 2): (953, 1331, 46, 295), (1, 0): (154, 530, 365, 614),
    (1, 1): (553, 931, 365, 614), (1, 2): (953, 1331, 365, 614),
}
TITLES = ["Original image & annotation", "Single-frame result & response",
          "Temporal result & response"]
ROW_LABELS = {0: "Recovering a\nmissed target",
              1: "Suppressing\nbackground\nfalse alarms"}
# burned-in score captions (row, col) -> value; cover rect in crop px coords
SCORES = {(0, 1): "0.071", (0, 2): "0.635", (1, 1): "0.316", (1, 2): "0.009"}
CHIP = dict(x0=236, y0=213, x1=373, y1=249)   # covers the Chinese caption

# colorbar: inner bar extent and tick-label centers measured on the source
CB_INNER = (135, 549)                          # top-down px
CB_TICKS = [("0.00", 550.5), ("0.01", 467.5), ("0.10", 364.5), ("1.00", 134.0)]


def setup_fonts():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })


def fy(y_px):
    return 1.0 - y_px / SH


def sample_cmap():
    """Column-averaged colors of the source colorbar, ordered bottom -> top."""
    a = np.asarray(Image.open(SRC_IMG).convert("RGB")).astype(float)
    grad = a[CB_INNER[0]:CB_INNER[1] + 1, 1362:1374].mean(axis=1)  # top-down
    return grad[::-1].astype(np.uint8)                             # bottom-up


def make_fig4():
    setup_fonts()
    fig = plt.figure(figsize=FIGSIZE)

    # ---- panels ---------------------------------------------------------------
    for (r, c), (x0, x1, y0, y1) in PANELS.items():
        ax = fig.add_axes([x0 / SW, fy(y1), (x1 - x0) / SW, (y1 - y0) / SH])
        ax.imshow(plt.imread(str(SRC / f"row{r + 1}_col{c + 1}.png")),
                  interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.7); s.set_color("#BBBBBB")
        if (r, c) in SCORES:                       # cover + English caption
            ax.add_patch(FancyBboxPatch(
                (CHIP["x0"], CHIP["y0"]), CHIP["x1"] - CHIP["x0"],
                CHIP["y1"] - CHIP["y0"],
                boxstyle="round,pad=0,rounding_size=3",
                fc="black", alpha=0.9, ec="none", zorder=4))
            ax.text((CHIP["x0"] + CHIP["x1"]) / 2, (CHIP["y0"] + CHIP["y1"]) / 2,
                    f"Score: {SCORES[(r, c)]}", ha="center", va="center",
                    fontsize=6, color="white", zorder=5)

    # ---- column titles (top row only, as in the source) -----------------------
    yc = fy(23.5)
    for c, t in enumerate(TITLES):
        xs = [v[0] for k, v in PANELS.items() if k[1] == c]
        xe = [v[1] for k, v in PANELS.items() if k[1] == c]
        fig.text((min(xs) + max(xe)) / 2 / SW, yc, t, ha="center", va="center",
                 fontsize=7, color=DARK)

    # ---- row labels ------------------------------------------------------------
    for r, lab in ROW_LABELS.items():
        ys = [v[2] for k, v in PANELS.items() if k[0] == r]
        ye = [v[3] for k, v in PANELS.items() if k[0] == r]
        fig.text(0.052, fy((min(ys) + max(ye)) / 2), lab, ha="center",
                 va="center", fontsize=7, color=DARK, linespacing=1.25)

    # ---- colorbar (rebuilt from sampled colors) --------------------------------
    y_top, y_bot = CB_INNER
    cax = fig.add_axes([1360 / SW, fy(550), 15.5 / SW, (y_bot - y_top + 1) / SH])
    cax.imshow(sample_cmap()[:, None, :], extent=(0, 1, 0, 1), aspect="auto",
           origin="lower", interpolation="bilinear")
    cax.set_xticks([])
    lo, hi = float(y_bot), float(y_top)          # lo = bottom px, hi = top px
    fracs = [min(max((lo - y) / (lo - hi), 0.0), 1.0) for _, y in CB_TICKS]
    cax.set_yticks(fracs)
    cax.set_yticklabels([t for t, _ in CB_TICKS])
    cax.yaxis.tick_right()
    cax.tick_params(axis="y", length=2.5, labelsize=6.5, pad=2)
    for s in cax.spines.values():
        s.set_linewidth(0.8); s.set_color("black")
    fig.text(0.986, fy((y_top + y_bot) / 2), "Detection response",
             rotation=90, ha="center", va="center", fontsize=7, color=DARK)

    fig.savefig(OUT / "fig4_cases_en.png", dpi=300)
    plt.close(fig)
    print("[ok] fig4_cases_en.png")


if __name__ == "__main__":
    make_fig4()
