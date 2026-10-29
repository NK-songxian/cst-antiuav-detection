# English rebuild of the Case-study figure (fig 6 in the CST manuscript).
# Source: /tmp/docx_media/image6.png (2x3 grid; rows = Case 1/2, cols = 1st/2nd/3rd
# observation). Photos were segmented by non-white density into fig6_src/ (3 px
# border trimmed). Only the text layer is re-rendered in English; photo content,
# boxes and layout geometry are unchanged from the source figure.
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
SRC = OUT / "fig6_src"

BLUE = "#2E5FA3"    # single-frame
ORANGE = "#E4622F"  # temporal
DARK = "#222222"
SEP = "#8A8A8A"

# ---- geometry measured on the source image (5331 x 1914 px) -------------------
SW, SH = 5331.0, 1914.0
W_IN = 17 / 2.54                 # 17 cm target width
FIGSIZE = (W_IN, W_IN * SH / SW)

# photo content bboxes (border trimmed), top-down px: (x0, x1, y0, y1)
PHOTOS = {
    ("case1", 0): (378, 1409, 184, 788),
    ("case1", 1): (2307, 3338, 184, 788),
    ("case1", 2): (4237, 5268, 184, 788),
    ("case2", 0): (477, 1310, 1159, 1763),
    ("case2", 1): (2406, 3239, 1159, 1763),
    ("case2", 2): (4336, 5169, 1159, 1763),
}
COL_CENTERS = [893.5, 2822.5, 4752.5]          # original photo bbox centers (px)
TITLES = ["1st observation", "2nd observation", "3rd observation"]
TITLE_BAND = {0: (56, 144), 1: (1030, 1118)}   # text bands (top-down px)
STATUS_BAND = {0: (795, 874), 1: (1770, 1849)}
# status per cell, read from the source: (single-frame, temporal)
STATUS = {
    ("case1", 0): ("miss", "detect"), ("case1", 1): ("miss", "detect"),
    ("case1", 2): ("miss", "detect"), ("case2", 0): ("miss", "miss"),
    ("case2", 1): ("miss", "detect"), ("case2", 2): ("miss", "miss"),
}
ROW_LABEL = {0: "Case 1", 1: "Case 2"}
LABEL_X = {0: (375 - 188) / SW, 1: (474 - 188) / SW}  # source-anchored label centers


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
    """top-down source px -> figure fraction (y flipped)."""
    return 1.0 - y_px / SH


def status_line(fig, x_frac, y_frac, single, temporal, fs=6.6):
    """One centered status line, two-colored: Single (blue) / Temporal (orange)."""
    parts = [(f"Single: {single}", BLUE), (" \u00b7 ", SEP),
             (f"Temporal: {temporal}", ORANGE)]
    texts = [fig.text(x_frac, y_frac, s, fontsize=fs, color=c,
                      ha="left", va="center", zorder=3) for s, c in parts]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [t.get_window_extent(renderer).width for t in texts]
    x0 = x_frac * fig.bbox.width - sum(widths) / 2.0
    for t, w in zip(texts, widths):
        t.set_position((x0 / fig.bbox.width, y_frac))
        x0 += w


def make_fig6():
    setup_fonts()
    fig = plt.figure(figsize=FIGSIZE)
    for (case, col), (x0, x1, y0, y1) in PHOTOS.items():
        ax = fig.add_axes([x0 / SW, fy(y1), (x1 - x0) / SW, (y1 - y0) / SH])
        ax.imshow(plt.imread(str(SRC / f"{case}_obs{col + 1}.png")),
                  interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.7); s.set_color("#BBBBBB")
    # column titles (repeated above both rows, as in the source)
    for row in (0, 1):
        yc = fy((TITLE_BAND[row][0] + TITLE_BAND[row][1]) / 2.0)
        for cx, t in zip(COL_CENTERS, TITLES):
            fig.text(cx / SW, yc, t, ha="center", va="center",
                     fontsize=8, color=DARK)
    # status line under each photo
    for (case, col), (x0, x1, y0, y1) in PHOTOS.items():
        row = 0 if case == "case1" else 1
        band = STATUS_BAND[row]
        yc = fy((band[0] + band[1]) / 2.0)
        cx = COL_CENTERS[col] / SW
        s, t = STATUS[(case, col)]
        status_line(fig, cx, yc, s, t)
    # row labels, vertically centered on each photo row
    for row, case in ((0, "case1"), (1, "case2")):
        ys = [v[2] for k, v in PHOTOS.items() if k[0] == case]
        ye = [v[3] for k, v in PHOTOS.items() if k[0] == case]
        yc = fy((min(ys) + max(ye)) / 2.0)
        fig.text(LABEL_X[row], yc, ROW_LABEL[row], ha="center", va="center",
                 fontsize=8, color=DARK)
    fig.savefig(OUT / "fig6_events_en.png", dpi=300)
    plt.close(fig)
    print("[ok] fig6_events_en.png")


if __name__ == "__main__":
    make_fig6()
