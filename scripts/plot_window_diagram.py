"""生成 README 4.2/4.3 的窗口查询示意图。

用法：
    python -m scripts.plot_window_diagram              # 中文版 → assets/window_query.png
    python -m scripts.plot_window_diagram --lang en    # 英文版 → assets/window_query_en.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "imu": "#ff9800",
    "eeg": "#4caf50",
    "state": "#e91e63",
    "temp": "#9c27b0",
    "query": "#d32f2f",
    "window": "#ffe082",
}

plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]

ROWS = {"temp": 0.0, "state": 1.0, "eeg": 2.0, "imu": 3.0}

TEXTS = {
    "zh": {
        "title": "查询 t = 0.333s（第 10 帧）时各 feature 的取数",
        "window_label": "window (-0.033, 0]",
        "query_label": "查询时刻 t = 0.333",
        "xlabel": "时间 (s)",
        "farther": "0.400（更远）",
        "legend_keep": "返回的读数 / 最近邻命中",
        "legend_skip": "未返回（窗口外 / 非最近）",
        "results": {
            "imu": "33 个读数 → [33, 6]",
            "eeg": "8 个读数 → [8, 8]",
            "state": "最近邻 → 命中 0.333 → [7]",
            "temp": "最近邻 → 取 t=0.300 → [1]",
        },
    },
    "en": {
        "title": "Readings returned per feature when querying t = 0.333s (frame 10)",
        "window_label": "window (-0.033, 0]",
        "query_label": "query time t = 0.333",
        "xlabel": "Time (s)",
        "farther": "0.400 (farther)",
        "legend_keep": "returned / nearest hit",
        "legend_skip": "not returned (outside window / not nearest)",
        "results": {
            "imu": "33 readings → [33, 6]",
            "eeg": "8 readings → [8, 8]",
            "state": "nearest → hit 0.333 → [7]",
            "temp": "nearest → picks t=0.300 → [1]",
        },
    },
}


def dots(ax, xs, y, color, inside):
    ax.scatter(
        xs,
        np.full(len(xs), y),
        s=22 if inside else 13,
        color=color,
        alpha=1.0 if inside else 0.35,
        zorder=3,
        linewidths=0,
    )


def ring(ax, x, y, color):
    ax.scatter([x], [y], s=110, facecolors="none", edgecolors=color,
               linewidths=1.6, zorder=4)


def main(lang: str = "zh"):
    t = TEXTS[lang]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0.29, 0.41)
    ax.set_ylim(-0.7, 3.9)

    # 每行基线
    for key, y in ROWS.items():
        ax.hlines(y, 0.29, 0.41, color=COLORS[key], alpha=0.2, lw=1, zorder=1)

    # window 阴影：只覆盖定义了 window 的 IMU / EEG 行
    y0, y1 = 1.35, 3.35
    frac = (y0 + 0.7) / 4.6, (y1 + 0.7) / 4.6
    ax.axvspan(0.300, 0.333, ymin=frac[0], ymax=frac[1],
               color=COLORS["window"], alpha=0.35, zorder=0)
    ax.text(0.3165, 3.62, t["window_label"], fontsize=9.5,
            ha="center", va="bottom", color="#8d6e63")

    # 查询时刻
    ax.axvline(0.333, color=COLORS["query"], ls=(0, (4, 3)), lw=1.2, zorder=5)
    ax.text(0.3342, 3.62, t["query_label"], fontsize=9.5,
            ha="left", va="bottom", color=COLORS["query"])

    # IMU 1000Hz：窗口内 33 个读数，窗外前后各几个（浅色）
    dots(ax, np.arange(0.301, 0.334, 0.001), ROWS["imu"], COLORS["imu"], True)
    dots(ax, [0.296, 0.297, 0.298, 0.299], ROWS["imu"], COLORS["imu"], False)
    dots(ax, [0.334, 0.335, 0.336, 0.337, 0.338], ROWS["imu"], COLORS["imu"], False)

    # EEG 256Hz：每帧写入 8 个读数（256 // 30）
    dots(ax, np.arange(78, 86) / 256, ROWS["eeg"], COLORS["eeg"], True)
    dots(ax, np.arange(75, 78) / 256, ROWS["eeg"], COLORS["eeg"], False)
    dots(ax, np.arange(86, 89) / 256, ROWS["eeg"], COLORS["eeg"], False)

    # State 30Hz：最近邻直接命中 0.333
    dots(ax, [0.3, 11 / 30], ROWS["state"], COLORS["state"], False)
    ring(ax, 1 / 3, ROWS["state"], COLORS["state"])

    # Temp 10Hz：0.333 处无采样点，最近邻取 0.300
    dots(ax, [0.4], ROWS["temp"], COLORS["temp"], False)
    ring(ax, 0.3, ROWS["temp"], COLORS["temp"])
    ax.text(0.4008, 0.25, t["farther"], fontsize=8.5, color="#999")

    # 右侧结果标注
    for key, y in ROWS.items():
        ax.text(0.414, y, t["results"][key], fontsize=10, color=COLORS[key],
                ha="left", va="center")

    # 图例：返回 vs 未返回
    legend_y = -0.55
    ax.scatter([0.292], [legend_y], s=22, color="#666", linewidths=0)
    ax.text(0.296, legend_y, t["legend_keep"], fontsize=8.5,
            va="center", color="#555")
    ax.scatter([0.337], [legend_y], s=13, color="#666", alpha=0.35, linewidths=0)
    ax.text(0.341, legend_y, t["legend_skip"], fontsize=8.5,
            va="center", color="#555")

    # 坐标轴
    freqs = {"imu": "1000Hz", "eeg": "256Hz", "state": "30Hz", "temp": "10Hz"}
    ax.set_yticks(list(ROWS.values()))
    ax.set_yticklabels([f"{k.upper()} {freqs[k]}" for k in ROWS], fontsize=10)
    for tick, key in zip(ax.get_yticklabels(), ROWS):
        tick.set_color(COLORS[key])
    ax.set_xticks([0.30, 0.32, 0.34, 0.36, 0.38, 0.40])
    ax.set_xticklabels(["0.30", "0.32", "0.34", "0.36", "0.38", "0.40"],
                       fontsize=9)
    ax.set_xlabel(t["xlabel"], fontsize=10)
    ax.set_title(t["title"], fontsize=12, pad=12)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)

    out = ROOT / "assets" / ("window_query.png" if lang == "zh"
                             else "window_query_en.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"written: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=["zh", "en"], default="zh",
                        help="figure language (default: zh)")
    main(parser.parse_args().lang)
