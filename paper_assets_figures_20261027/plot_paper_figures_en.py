# -*- coding: utf-8 -*-
"""
CST anti-UAV detection paper metric figures (fig2 / fig3 / fig5), English version.
Only the text language differs from the Chinese version; design and values are
unchanged. All annotated values come from verified paper values or source data:
  - Fig2  : dumbbell values are verified paper values (hard-coded; gains
            consistent with the two endpoints)
  - Fig3a : means are three-seed means (hard-coded); standard deviations from
            experiments/cst_gap_confirmation_20261019/analysis/new_metrics/
            test_audit/{N_current,N_tcs}_seed{0..2}/audit.json, three-seed
            sample standard deviation (ddof=1) of the conf_0.25
            background_frame_fp_rate
  - Fig3b : recomputed frame-by-frame at runtime from
            test_audit/{N_current,N_tcs}_seed{0..2}
            predictions_conf010.json + frame_stats.npz
            (frame-level BG-FP = frames with >=1 prediction box score >= tau
            among target-free frames / total target-free frames)
  - Fig3c : verified paper values (hard-coded)
  - Fig5  : verified paper values (hard-coded, denominator 178)
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from PIL import Image

# ----------------------------------------------------------------------------- paths
OUT_DIR = Path(__file__).resolve().parent
AUDIT_BASE = Path("/root/autodl-tmp/Ryan/cst_antiuav/experiments/"
                  "cst_gap_confirmation_20261019/analysis/new_metrics/test_audit")

# ----------------------------------------------------------------------------- shared style
BLUE = "#2E5FA3"      # single-frame
ORANGE = "#E4622F"    # temporal
GRAY_TXT = "#767676"  # auxiliary gray text (gains / means)
GRID_KW = dict(ls=(0, (2, 3)), lw=0.6, color="#D9D9D9", zorder=0)


def setup_fonts():
    """English version: DejaVu Sans only, no CJK font required."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })
    print("[font] Using font: DejaVu Sans")


def style_axes(ax, grid_axis="x"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(**{**GRID_KW, "axis": grid_axis})
    ax.set_axisbelow(True)


def legend_single_vs_temporal(fig, marker_line=True, loc_x=0.5):
    """One legend row on top: single-frame (blue circle) / temporal (orange square)."""
    if marker_line:
        handles = [Line2D([0], [0], color=BLUE, marker="o", ms=6, lw=1.6, label="Single-frame"),
                   Line2D([0], [0], color=ORANGE, marker="s", ms=6, lw=1.6, label="Temporal")]
    else:
        handles = [Line2D([0], [0], color=BLUE, marker="o", ms=6, ls="", label="Single-frame"),
                   Line2D([0], [0], color=ORANGE, marker="s", ms=6, ls="", label="Temporal")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(loc_x, 1.0), columnspacing=1.6, handletextpad=0.5)


# =============================================================================== fig2
ROWS2 = ["YOLO11n (Seed 0)", "YOLO11n (Seed 1)", "YOLO11n (Seed 2)", "YOLO11s (Seed 0)"]
FIG2 = {
    "mAP50": {
        "cur": [61.17, 58.49, 58.38, 61.78],
        "tcs": [74.73, 72.55, 75.36, 77.12],
        "gain": ["+13.56 pp", "+14.06 pp", "+16.98 pp", "+15.34 pp"],
        "xlim": (50, 80),
        "title": "(a) mAP50",
    },
    "mAP50-95": {
        "cur": [33.52, 31.97, 32.66, 34.74],
        "tcs": [39.26, 39.25, 40.28, 41.40],
        "gain": ["+5.74 pp", "+7.28 pp", "+7.62 pp", "+6.66 pp"],
        "xlim": (30, 43),
        "title": "(b) mAP50–95",
    },
}


def make_fig2():
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.0))
    fig.subplots_adjust(left=0.235, right=0.905, top=0.775, bottom=0.15, wspace=0.36)
    y = [3, 2, 1, 0]
    for ax, (key, d) in zip(axes, FIG2.items()):
        cur, tcs = d["cur"], d["tcs"]
        ax.set_xlim(*d["xlim"])
        ax.set_ylim(-0.7, 3.7)
        ax.axhline(0.5, color="#C8C8C8", ls=(0, (4, 3)), lw=0.9, zorder=1)  # 11n/11s separator
        style_axes(ax, grid_axis="x")
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(ROWS2)
        else:
            ax.tick_params(labelleft=False)  # right panel shares the left row labels
        ax.set_xlabel("Accuracy (%)")
        ax.set_title(d["title"], loc="left", fontweight="bold", fontsize=10, pad=6)
        for yi, c0, c1 in zip(y, cur, tcs):
            ax.plot([c0, c1], [yi, yi], color="#BFBFBF", lw=2.4, zorder=2,
                    solid_capstyle="round")
        ax.scatter(cur, y, s=60, c=BLUE, marker="o", zorder=3)
        ax.scatter(tcs, y, s=60, c=ORANGE, marker="s", zorder=3)
        # value labels: blue below its point, orange above its point, white bbox
        for yi, v in zip(y, cur):
            ax.annotate(f"{v:.2f}", (v, yi), textcoords="offset points", xytext=(0, -8),
                        ha="center", va="top", fontsize=8.5, color=BLUE,
                        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=BLUE,
                                  lw=0.4, alpha=0.95))
        for yi, v in zip(y, tcs):
            ax.annotate(f"{v:.2f}", (v, yi), textcoords="offset points", xytext=(0, 8),
                        ha="center", va="bottom", fontsize=8.5, color=ORANGE,
                        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=ORANGE,
                                  lw=0.4, alpha=0.95))
        # gain at the end of each row (small gray text outside the panel)
        bt = blended_transform_factory(ax.transAxes, ax.transData)
        for yi, g in zip(y, d["gain"]):
            ax.text(1.02, yi, g, transform=bt, ha="left", va="center",
                    fontsize=8, color=GRAY_TXT)
    legend_single_vs_temporal(fig, marker_line=False, loc_x=0.5)
    fig.savefig(OUT_DIR / "fig2_input_contrast_en.png")
    plt.close(fig)
    print("[ok] fig2_input_contrast_en.png")


# =============================================================================== fig3
# Fig3a: mean (three-seed mean, same as the paper) + sd (recomputed from audit.json, ddof=1)
FIG3A_ROWS = ["All background", "Out-of-view (OV)", "Occlusion (OC)",
              "Thermal crosstalk (TC)", "Dynamic clutter (DBC)", "Complex dyn. bg (CDB)"]
FIG3A = {  # row: (cur_mean, cur_sd, tcs_mean, tcs_sd)
    "All background":         (17.49, 2.44, 15.41, 4.77),
    "Out-of-view (OV)":       (27.34, 4.81, 20.94, 5.67),
    "Occlusion (OC)":         (8.08, 0.31, 10.16, 4.10),
    "Thermal crosstalk (TC)": (7.63, 0.67, 9.26, 3.75),
    "Dynamic clutter (DBC)":  (7.26, 0.78, 10.11, 3.53),
    "Complex dyn. bg (CDB)":  (8.83, 1.66, 9.15, 3.51),
}
FIG3C = [  # (ov_frames, total) false-alarm frame composition of the temporal models
    (345, 542), (227, 323), (217, 333),
]


def compute_bgfp_curves():
    """Fig3b: recompute the threshold vs. background FP rate curves (mean +/- cross-seed sd)."""
    taus = np.round(np.arange(0.10, 0.501, 0.05), 2)
    out = {}
    for key, prefix in [("current", "N_current"), ("tcs", "N_tcs")]:
        per_seed = []
        for s in range(3):
            d = AUDIT_BASE / f"{prefix}_seed{s}"
            z = np.load(d / "frame_stats.npz", allow_pickle=True)
            stems, tgt = z["frame_stems"], z["target_image_idx"]
            has_obj = np.zeros(len(stems), bool)
            has_obj[tgt] = True
            bg_names = stems[~has_obj]
            preds = json.load(open(d / "predictions_conf010.json"))
            maxconf = np.array([
                max(preds[n]["conf"]) if n in preds and preds[n]["conf"] else -1.0
                for n in bg_names
            ])
            rates = (maxconf[None, :] >= taus[:, None]).mean(axis=1) * 100.0
            per_seed.append(rates)
        per_seed = np.asarray(per_seed)          # (3, n_taus)
        out[key] = (per_seed.mean(axis=0), per_seed.std(axis=0, ddof=1))
    return taus, out


def make_fig3(taus, curves):
    fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.85),
                             gridspec_kw=dict(width_ratios=[1.16, 1.04, 1.09]))
    fig.subplots_adjust(left=0.235, right=0.975, top=0.78, bottom=0.20, wspace=0.50)
    ax_a, ax_b, ax_c = axes

    # ---- (a) attribute-stratified dumbbell + error bars -------------------------
    n = len(FIG3A_ROWS)
    ypos = np.arange(n)[::-1]                    # top to bottom 5..0
    cur_m = np.array([FIG3A[r][0] for r in FIG3A_ROWS])
    cur_sd = np.array([FIG3A[r][1] for r in FIG3A_ROWS])
    tcs_m = np.array([FIG3A[r][2] for r in FIG3A_ROWS])
    tcs_sd = np.array([FIG3A[r][3] for r in FIG3A_ROWS])
    ov_y = ypos[FIG3A_ROWS.index("Out-of-view (OV)")]
    ax_a.axhspan(ov_y - 0.42, ov_y + 0.42, color="#DCE9F8", zorder=0)  # light-blue OV row
    for yi, c0, c1 in zip(ypos, cur_m, tcs_m):
        ax_a.plot([c0, c1], [yi, yi], color="#BFBFBF", lw=1.8, zorder=2,
                  solid_capstyle="round")
    ax_a.errorbar(cur_m, ypos, xerr=cur_sd, fmt="o", color=BLUE, ms=5.5,
                  capsize=2.5, elinewidth=1.1, capthick=1.0, zorder=3)
    ax_a.errorbar(tcs_m, ypos, xerr=tcs_sd, fmt="s", color=ORANGE, ms=5.5,
                  capsize=2.5, elinewidth=1.1, capthick=1.0, zorder=3)
    # value labels: blue label at the upper-outer corner of the blue point, orange
    # likewise (above the error-bar line); rows where the blue point lies right of
    # the orange one (All background / Out-of-view) are mirrored to avoid collisions
    for yi, m0, m1 in zip(ypos, cur_m, tcs_m):
        if m0 > m1:      # blue on the right: blue label upper-right, orange upper-left
            off0, ha0, off1, ha1 = (6, 7), "left", (-6, 7), "right"
        else:            # default: blue upper-left, orange upper-right
            off0, ha0, off1, ha1 = (-6, 7), "right", (6, 7), "left"
        ax_a.annotate(f"{m0:.2f}", (m0, yi), textcoords="offset points",
                      xytext=off0, ha=ha0, va="bottom", fontsize=8.5,
                      color=BLUE, zorder=4,
                      bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                                alpha=0.9))
        ax_a.annotate(f"{m1:.2f}", (m1, yi), textcoords="offset points",
                      xytext=off1, ha=ha1, va="bottom", fontsize=8.5,
                      color=ORANGE, zorder=4,
                      bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                                alpha=0.9))
    ax_a.set_xlim(0, 40)
    ax_a.set_ylim(-0.6, n - 0.05)
    ax_a.set_yticks(ypos)
    ax_a.set_yticklabels(FIG3A_ROWS)
    ax_a.set_xlabel("Background FP rate (%)")
    ax_a.set_title("(a) Per-attribute (τ = 0.25)", loc="left", fontweight="bold",
                   fontsize=9, pad=6)
    style_axes(ax_a, grid_axis="x")

    # ---- (b) threshold sweep curves ---------------------------------------------
    cur_mn, cur_sd_ = curves["current"]
    tcs_mn, tcs_sd_ = curves["tcs"]
    ax_b.fill_between(taus, cur_mn - cur_sd_, cur_mn + cur_sd_, color=BLUE,
                      alpha=0.16, lw=0)
    ax_b.fill_between(taus, tcs_mn - tcs_sd_, tcs_mn + tcs_sd_, color=ORANGE,
                      alpha=0.16, lw=0)
    ax_b.plot(taus, cur_mn, color=BLUE, lw=1.6, marker="o", ms=4, zorder=3)
    ax_b.plot(taus, tcs_mn, color=ORANGE, lw=1.6, marker="s", ms=4, zorder=3)
    ax_b.axvline(0.25, color="#9A9A9A", ls=(0, (4, 3)), lw=1.0, zorder=2)
    ax_b.text(0.258, 0.965, "τ = 0.25", transform=ax_b.get_xaxis_transform(),
              ha="left", va="top", fontsize=7.8, color=GRAY_TXT)
    ax_b.set_xlim(0.08, 0.52)
    ax_b.set_ylim(0, 33)
    ax_b.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax_b.set_xticklabels(["0.10", "0.20", "0.30", "0.40", "0.50"], fontsize=8)
    ax_b.set_yticks([0, 10, 20, 30])
    ax_b.set_xlabel("Confidence threshold τ")
    ax_b.set_ylabel("Background FP rate (%)")
    ax_b.set_title("(b) Threshold sweep", loc="left", fontweight="bold",
                   fontsize=9, pad=6)
    style_axes(ax_b, grid_axis="y")

    # ---- (c) temporal false-alarm frame composition -----------------------------
    ov_pct = np.array([f / t * 100 for f, t in FIG3C])
    non_pct = 100 - ov_pct
    y = np.arange(3)[::-1]                       # seed 0 on top
    labels = [f"{f}/{t}\n({p:.2f}%)" for (f, t), p in zip(FIG3C, ov_pct)]
    ax_c.barh(y, ov_pct, height=0.52, color=BLUE, zorder=2)
    ax_c.barh(y, non_pct, left=ov_pct, height=0.52, color="#C7D3E6", zorder=2)
    for yi, wpc, lab in zip(y, ov_pct, labels):
        ax_c.text(wpc / 2, yi, lab, ha="center", va="center", fontsize=7.5,
                  color="white", zorder=3)
    ax_c.set_xlim(0, 100)
    ax_c.set_ylim(-0.55, 2.55)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(["Seed 0", "Seed 1", "Seed 2"])
    ax_c.set_xticks([0, 25, 50, 75, 100])
    ax_c.set_xlabel("Share of false-alarm\nbackground frames (%)", fontsize=9)
    ax_c.set_title("(c) FP composition", loc="left", fontweight="bold",
                   fontsize=9, pad=6)
    style_axes(ax_c, grid_axis="x")
    # OV / non-OV legend at the top-right of the whole figure (inside the panel it
    # would cover the white value labels on the bars)
    fig.legend(handles=[Patch(fc=BLUE, label="OV frames"),
                        Patch(fc="#C7D3E6", label="Non-OV")],
               loc="upper right", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.995, 1.0), columnspacing=1.0, handletextpad=0.5)

    legend_single_vs_temporal(fig, marker_line=True, loc_x=0.30)
    fig.savefig(OUT_DIR / "fig3_false_alarm_en.png")
    plt.close(fig)
    print("[ok] fig3_false_alarm_en.png")


# =============================================================================== fig5
FIG5 = {
    "a": {
        "cur": [37, 38, 36],
        "tcs": [52, 57, 58],
        "title": "(a) First-observation detection ($R_1$)",
        "ylim": (0, 70),
        "yticks": [0, 10, 20, 30, 40, 50, 60, 70],
    },
    "b": {
        "cur": [69, 56, 62],
        "tcs": [90, 87, 100],
        "title": "(b) ≥1 detection in first three ($R_3$)",
        "ylim": (0, 115),
        "yticks": [0, 20, 40, 60, 80, 100],
    },
}


def make_fig5():
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.05))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.775, bottom=0.135, wspace=0.42)
    x = np.arange(3)
    w = 0.36
    for ax, key in zip(axes, ["a", "b"]):
        d = FIG5[key]
        cur, tcs = d["cur"], d["tcs"]
        ax.bar(x - 0.19, cur, width=w, color=BLUE, zorder=2)
        ax.bar(x + 0.19, tcs, width=w, color=ORANGE, zorder=2)
        # n/178 labels on top of each bar, plain black text, no bbox
        for xi, v in zip(x - 0.19, cur):
            ax.text(xi, v + 1.5, f"{v}/178", ha="center", va="bottom",
                    fontsize=8.5, color="black", zorder=3)
        for xi, v in zip(x + 0.19, tcs):
            ax.text(xi, v + 1.5, f"{v}/178", ha="center", va="bottom",
                    fontsize=8.5, color="black", zorder=3)
        # three-seed means are reported in the text and Table 5; no mean line here
        ax.set_xlim(-0.75, 2.75)
        ax.set_ylim(*d["ylim"])
        ax.set_yticks(d["yticks"])
        ax.set_xticks(x)
        ax.set_xticklabels(["Seed 0", "Seed 1", "Seed 2"])
        ax.set_ylabel("Detected frames")
        ax.set_title(d["title"], loc="left", fontweight="bold", fontsize=9.5, pad=6)
        style_axes(ax, grid_axis="y")
    legend_single_vs_temporal(fig, marker_line=False, loc_x=0.5)
    fig.savefig(OUT_DIR / "fig5_reappearance_en.png")
    plt.close(fig)
    print("[ok] fig5_reappearance_en.png")


# =============================================================================== self-check
def self_check(taus, curves):
    idx = int(np.where(np.isclose(taus, 0.25))[0][0])
    cur25 = curves["current"][0][idx]
    tcs25 = curves["tcs"][0][idx]
    print(f"[check] fig3b tau=0.25 recomputed: single-frame {cur25:.2f}% (ref 17.49, diff {cur25 - 17.49:+.4f}); "
          f"temporal {tcs25:.2f}% (ref 15.41, diff {tcs25 - 15.41:+.4f})")
    assert abs(cur25 - 17.49) < 0.05, "single-frame tau=0.25 check failed"
    assert abs(tcs25 - 15.41) < 0.05, "temporal tau=0.25 check failed"
    for name in ["fig2_input_contrast_en.png", "fig3_false_alarm_en.png",
                 "fig5_reappearance_en.png"]:
        p = OUT_DIR / name
        im = Image.open(p)
        g = np.asarray(im.convert("L"))
        nonwhite = float((g < 245).mean())
        print(f"[check] {name}: {im.size[0]}x{im.size[1]} px, "
              f"mode {im.mode}, non-white pixel share {nonwhite:.1%}")
        assert im.size[0] >= 1900 and im.size[1] >= 700, f"{name} unexpected size"
        assert nonwhite > 0.02, f"{name} looks blank"


if __name__ == "__main__":
    setup_fonts()
    taus, curves = compute_bgfp_curves()
    print("[data] fig3b recomputed curves:")
    print("  taus          :", np.round(taus, 2).tolist())
    print("  single mean   :", np.round(curves["current"][0], 2).tolist())
    print("  single sd     :", np.round(curves["current"][1], 2).tolist())
    print("  temporal mean :", np.round(curves["tcs"][0], 2).tolist())
    print("  temporal sd   :", np.round(curves["tcs"][1], 2).tolist())
    make_fig2()
    make_fig3(taus, curves)
    make_fig5()
    self_check(taus, curves)
    print("[done] all figures generated")
