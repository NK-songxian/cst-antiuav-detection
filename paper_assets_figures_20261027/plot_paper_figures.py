# -*- coding: utf-8 -*-
"""
CST 反无人机检测论文指标图重绘（图2 / 图3 / 图5）。
仅改视觉呈现；所有标注数值均来自已验证的论文数值或实验源数据：
  - 图2   : 哑铃图数据为论文已验证数值（硬编码，增益与两端差值一致）
  - 图3a  : 均值为三种子均值（硬编码）；标准差取自
            experiments/cst_gap_confirmation_20261019/analysis/new_metrics/
            test_audit/{N_current,N_tcs}_seed{0..2}/audit.json 三种子
            conf_0.25 background_frame_fp_rate 的样本标准差（ddof=1）
  - 图3b  : 运行时从 test_audit/{N_current,N_tcs}_seed{0..2} 的
            predictions_conf010.json + frame_stats.npz 逐帧重算
            （帧级 BG-FP = 无目标帧中 >=1 个预测框分数 >= tau 的帧数 / 无目标帧总数）
  - 图3c  : 论文已验证数值（硬编码）
  - 图5   : 论文已验证数值（硬编码，分母 178）
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from PIL import Image

# ----------------------------------------------------------------------------- 路径
OUT_DIR = Path(__file__).resolve().parent
AUDIT_BASE = Path("/root/autodl-tmp/Ryan/cst_antiuav/experiments/"
                  "cst_gap_confirmation_20261019/analysis/new_metrics/test_audit")

# ----------------------------------------------------------------------------- 统一风格
BLUE = "#2E5FA3"      # 单帧
ORANGE = "#E4622F"    # 时序
GRAY_TXT = "#767676"  # 增益/均值等辅助灰字
GRID_KW = dict(ls=(0, (2, 3)), lw=0.6, color="#D9D9D9", zorder=0)


def setup_fonts():
    """优先系统 CJK 字体，否则注册本机已下载的 Noto Sans CJK SC；找不到则报错退出。"""
    cjk_candidates = ["Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC",
                      "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "SimHei",
                      "Microsoft YaHei", "PingFang SC"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    if not any(c in installed for c in cjk_candidates):
        local_fonts = sorted(Path("/root/.local/share/fonts").glob("NotoSansCJK*.otf"))
        if not local_fonts:
            raise RuntimeError("未找到任何可用中文字体（系统与 ~/.local/share/fonts 均无），"
                               "按要求报告而非瞎编。")
        for p in local_fonts:
            font_manager.fontManager.addfont(str(p))
        installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [c for c in cjk_candidates if c in installed][0]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [chosen, "DejaVu Sans"],
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
    print(f"[font] 使用中文字体: {chosen}")
    return chosen


def style_axes(ax, grid_axis="x"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(**{**GRID_KW, "axis": grid_axis})
    ax.set_axisbelow(True)


def legend_single_vs_temporal(fig, marker_line=True, loc_x=0.5):
    """面板顶部一行图例：单帧（蓝圆）/ 时序（橙方）。"""
    if marker_line:
        handles = [Line2D([0], [0], color=BLUE, marker="o", ms=6, lw=1.6, label="单帧"),
                   Line2D([0], [0], color=ORANGE, marker="s", ms=6, lw=1.6, label="时序")]
    else:
        handles = [Line2D([0], [0], color=BLUE, marker="o", ms=6, ls="", label="单帧"),
                   Line2D([0], [0], color=ORANGE, marker="s", ms=6, ls="", label="时序")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(loc_x, 1.0), columnspacing=1.6, handletextpad=0.5)


# =============================================================================== 图2
ROWS2 = ["YOLO11n·种子0", "YOLO11n·种子1", "YOLO11n·种子2", "YOLO11s·种子0"]
FIG2 = {
    "mAP50": {
        "cur": [61.17, 58.49, 58.38, 61.78],
        "tcs": [74.73, 72.55, 75.36, 77.12],
        "gain": ["+13.56", "+14.06", "+16.98", "+15.34"],
        "xlim": (50, 80),
        "title": "(a) mAP50",
    },
    "mAP50-95": {
        "cur": [33.52, 31.97, 32.66, 34.74],
        "tcs": [39.26, 39.25, 40.28, 41.40],
        "gain": ["+5.74", "+7.28", "+7.62", "+6.66"],
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
        ax.axhline(0.5, color="#C8C8C8", ls=(0, (4, 3)), lw=0.9, zorder=1)  # 11n/11s 分隔
        style_axes(ax, grid_axis="x")
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(ROWS2)
        else:
            ax.tick_params(labelleft=False)  # 右面板共享左侧行标签
        ax.set_xlabel("精度 (%)")
        ax.set_title(d["title"], loc="left", fontweight="bold", fontsize=10, pad=6)
        for yi, c0, c1 in zip(y, cur, tcs):
            ax.plot([c0, c1], [yi, yi], color="#BFBFBF", lw=2.4, zorder=2,
                    solid_capstyle="round")
        ax.scatter(cur, y, s=60, c=BLUE, marker="o", zorder=3)
        ax.scatter(tcs, y, s=60, c=ORANGE, marker="s", zorder=3)
        # 数值标注：蓝值点下、橙值点上，白底细描边
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
        # 行末增益（面板右外侧小灰字）
        bt = blended_transform_factory(ax.transAxes, ax.transData)
        for yi, g in zip(y, d["gain"]):
            ax.text(1.02, yi, g, transform=bt, ha="left", va="center",
                    fontsize=8, color=GRAY_TXT)
    legend_single_vs_temporal(fig, marker_line=False, loc_x=0.5)
    fig.savefig(OUT_DIR / "fig2_input_contrast.png")
    plt.close(fig)
    print("[ok] fig2_input_contrast.png")


# =============================================================================== 图3
# 图3a：均值（三种子均值，与论文一致）+ 标准差（audit.json 三种子重算，ddof=1）
FIG3A_ROWS = ["全部背景", "视外OV", "遮挡OC", "热交叉TC", "动态杂波DBC", "复杂动态背景CDB"]
FIG3A = {  # row: (cur_mean, cur_sd, tcs_mean, tcs_sd)
    "全部背景":       (17.49, 2.44, 15.41, 4.77),
    "视外OV":         (27.34, 4.81, 20.94, 5.67),
    "遮挡OC":         (8.08, 0.31, 10.16, 4.10),
    "热交叉TC":       (7.63, 0.67, 9.26, 3.75),
    "动态杂波DBC":    (7.26, 0.78, 10.11, 3.53),
    "复杂动态背景CDB": (8.83, 1.66, 9.15, 3.51),
}
FIG3C = [  # (ov_frames, total) 时序模型误报帧构成
    (345, 542), (227, 323), (217, 333),
]


def compute_bgfp_curves():
    """图3b：逐帧重算阈值-背景误报率曲线（均值±跨种子样本标准差）。"""
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
                             gridspec_kw=dict(width_ratios=[1.22, 0.95, 1.12]))
    fig.subplots_adjust(left=0.165, right=0.985, top=0.78, bottom=0.20, wspace=0.50)
    ax_a, ax_b, ax_c = axes

    # ---- (a) 属性分层哑铃 + 误差棒 ---------------------------------------------
    n = len(FIG3A_ROWS)
    ypos = np.arange(n)[::-1]                    # 从上到下 5..0
    cur_m = np.array([FIG3A[r][0] for r in FIG3A_ROWS])
    cur_sd = np.array([FIG3A[r][1] for r in FIG3A_ROWS])
    tcs_m = np.array([FIG3A[r][2] for r in FIG3A_ROWS])
    tcs_sd = np.array([FIG3A[r][3] for r in FIG3A_ROWS])
    ov_y = ypos[FIG3A_ROWS.index("视外OV")]
    ax_a.axhspan(ov_y - 0.42, ov_y + 0.42, color="#DCE9F8", zorder=0)  # OV 行浅蓝高亮
    for yi, c0, c1 in zip(ypos, cur_m, tcs_m):
        ax_a.plot([c0, c1], [yi, yi], color="#BFBFBF", lw=1.8, zorder=2,
                  solid_capstyle="round")
    ax_a.errorbar(cur_m, ypos, xerr=cur_sd, fmt="o", color=BLUE, ms=5.5,
                  capsize=2.5, elinewidth=1.1, capthick=1.0, zorder=3)
    ax_a.errorbar(tcs_m, ypos, xerr=tcs_sd, fmt="s", color=ORANGE, ms=5.5,
                  capsize=2.5, elinewidth=1.1, capthick=1.0, zorder=3)
    # 数值标签：蓝值在蓝点外侧上角、橙值在橙点外侧上角（误差棒线之上，不压误差棒）；
    # 蓝点位于橙点右侧的行（全部背景/视外OV）反向放置，避免两标签相撞
    for yi, m0, m1 in zip(ypos, cur_m, tcs_m):
        if m0 > m1:      # 蓝点在右：蓝值放右上、橙值放左上
            off0, ha0, off1, ha1 = (6, 7), "left", (-6, 7), "right"
        else:            # 常规：蓝值放左上、橙值放右上
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
    ax_a.set_xlabel("背景帧误报率 (%)")
    ax_a.set_title("(a) 属性分层（τ = 0.25）", loc="left", fontweight="bold",
                   fontsize=10, pad=6)
    style_axes(ax_a, grid_axis="x")

    # ---- (b) 阈值敏感性曲线 ----------------------------------------------------
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
    ax_b.set_xticklabels(["0.10", "0.20", "0.30", "0.40", "0.50"])
    ax_b.set_yticks([0, 10, 20, 30])
    ax_b.set_xlabel("置信度阈值 τ")
    ax_b.set_ylabel("背景帧误报率 (%)")
    ax_b.set_title("(b) 置信度阈值敏感性", loc="left", fontweight="bold",
                   fontsize=10, pad=6)
    style_axes(ax_b, grid_axis="y")

    # ---- (c) 时序误报帧构成 ----------------------------------------------------
    ov_pct = np.array([f / t * 100 for f, t in FIG3C])
    non_pct = 100 - ov_pct
    y = np.arange(3)[::-1]                       # 种子0 在上
    labels = [f"{f}/{t} ({p:.2f}%)" for (f, t), p in zip(FIG3C, ov_pct)]
    ax_c.barh(y, ov_pct, height=0.52, color=BLUE, zorder=2)
    ax_c.barh(y, non_pct, left=ov_pct, height=0.52, color="#C7D3E6", zorder=2)
    for yi, wpc, lab in zip(y, ov_pct, labels):
        ax_c.text(wpc / 2, yi, lab, ha="center", va="center", fontsize=7.5,
                  color="white", zorder=3)
    ax_c.set_xlim(0, 100)
    ax_c.set_ylim(-0.55, 2.55)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(["种子0", "种子1", "种子2"])
    ax_c.set_xticks([0, 25, 50, 75, 100])
    ax_c.set_xlabel("占误报背景帧比例 (%)")
    ax_c.set_title("(c) 时序误报帧构成", loc="left", fontweight="bold",
                   fontsize=10, pad=6)
    style_axes(ax_c, grid_axis="x")
    # OV / 非 OV 图例放整图顶部右侧（面板内会遮挡条上白字数值）
    fig.legend(handles=[Patch(fc=BLUE, label="视外（OV）帧"),
                        Patch(fc="#C7D3E6", label="非 OV 帧")],
               loc="upper right", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.995, 1.0), columnspacing=1.0, handletextpad=0.5)

    legend_single_vs_temporal(fig, marker_line=True, loc_x=0.30)
    fig.savefig(OUT_DIR / "fig3_false_alarm.png")
    plt.close(fig)
    print("[ok] fig3_false_alarm.png")


# =============================================================================== 图5
FIG5 = {
    "a": {
        "cur": [37, 38, 36],
        "tcs": [52, 57, 58],
        "title": "(a) 首次观测正确检出（$R_1$）",
        "ylim": (0, 70),
        "yticks": [0, 10, 20, 30, 40, 50, 60, 70],
    },
    "b": {
        "cur": [69, 56, 62],
        "tcs": [90, 87, 100],
        "title": "(b) 前三次内至少一次正确检出（$R_3$）",
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
        # n/178 柱顶标注：统一放各自柱顶正上方黑字，无底框
        for xi, v in zip(x - 0.19, cur):
            ax.text(xi, v + 1.5, f"{v}/178", ha="center", va="bottom",
                    fontsize=8.5, color="black", zorder=3)
        for xi, v in zip(x + 0.19, tcs):
            ax.text(xi, v + 1.5, f"{v}/178", ha="center", va="bottom",
                    fontsize=8.5, color="black", zorder=3)
        # 三种子均值见正文与表5，图中不再画均值线
        ax.set_xlim(-0.75, 2.75)
        ax.set_ylim(*d["ylim"])
        ax.set_yticks(d["yticks"])
        ax.set_xticks(x)
        ax.set_xticklabels(["种子0", "种子1", "种子2"])
        ax.set_ylabel("检出帧数")
        ax.set_title(d["title"], loc="left", fontweight="bold", fontsize=10, pad=6)
        style_axes(ax, grid_axis="y")
    legend_single_vs_temporal(fig, marker_line=False, loc_x=0.5)
    fig.savefig(OUT_DIR / "fig5_reappearance.png")
    plt.close(fig)
    print("[ok] fig5_reappearance.png")


# =============================================================================== 自检
def self_check(taus, curves):
    idx = int(np.where(np.isclose(taus, 0.25))[0][0])
    cur25 = curves["current"][0][idx]
    tcs25 = curves["tcs"][0][idx]
    print(f"[check] 图3b τ=0.25 重算: 单帧 {cur25:.2f}% (参考 17.49, 偏差 {cur25 - 17.49:+.4f}); "
          f"时序 {tcs25:.2f}% (参考 15.41, 偏差 {tcs25 - 15.41:+.4f})")
    assert abs(cur25 - 17.49) < 0.05, "单帧 τ=0.25 校验失败"
    assert abs(tcs25 - 15.41) < 0.05, "时序 τ=0.25 校验失败"
    for name in ["fig2_input_contrast.png", "fig3_false_alarm.png",
                 "fig5_reappearance.png"]:
        p = OUT_DIR / name
        im = Image.open(p)
        g = np.asarray(im.convert("L"))
        nonwhite = float((g < 245).mean())
        print(f"[check] {name}: {im.size[0]}x{im.size[1]} px, "
              f"模式 {im.mode}, 非白像素占比 {nonwhite:.1%}")
        assert im.size[0] >= 1900 and im.size[1] >= 700, f"{name} 尺寸异常"
        assert nonwhite > 0.02, f"{name} 疑似空白图"


if __name__ == "__main__":
    setup_fonts()
    taus, curves = compute_bgfp_curves()
    print("[data] 图3b 重算结果:")
    print("  taus      :", np.round(taus, 2).tolist())
    print("  单帧 mean :", np.round(curves["current"][0], 2).tolist())
    print("  单帧 sd   :", np.round(curves["current"][1], 2).tolist())
    print("  时序 mean :", np.round(curves["tcs"][0], 2).tolist())
    print("  时序 sd   :", np.round(curves["tcs"][1], 2).tolist())
    make_fig2()
    make_fig3(taus, curves)
    make_fig5()
    self_check(taus, curves)
    print("[done] 全部完成")
