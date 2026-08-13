#!/usr/bin/env python
"""Interactive multi-frequency dataset visualizer.

Controls:
    Space — play / pause
    ← →  — step frame
    ↑ ↓  — speed up / down
    w s  — widen / shrink window
    q    — quit
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from mf_lerobot import MultiFrequencyLeRobotDataset

# ── Dark theme ──
plt.style.use("dark_background")
matplotlib.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
    "axes.edgecolor": "#444", "grid.alpha": 0.15,
})

root = Path(__file__).parent.parent / "data" / "demo"
ds = MultiFrequencyLeRobotDataset(repo_id="robot_demo", root=root, video_backend="pyav")

CAMERA_KEYS = sorted(k for k, ft in ds.meta.features.items()
                     if ft.get("dtype") in ("video", "image"))
DATA_KEYS = [k for k in ds.meta.data_keys if k not in CAMERA_KEYS]

# Per-feature color palette
FEATURE_COLORS = {
    "imu": plt.cm.plasma, "eeg": plt.cm.viridis,
    "emg": plt.cm.magma, "temperature": plt.cm.inferno,
    "state": plt.cm.cool, "action": plt.cm.autumn,
}


def _color_map(key):
    for tag, cmap in FEATURE_COLORS.items():
        if tag in key:
            return cmap
    return plt.cm.tab10


class Visualizer:
    def __init__(self, ep_idx=0, play_fps=10):
        self.ep_idx = ep_idx
        self.play_fps = play_fps
        self.interval = 1000 / play_fps
        self.window_size = 60

        ep_arr = np.array(ds.hf_dataset["episode_index"])
        self.global_indices = np.where(ep_arr == ep_idx)[0]
        self.n_frames = len(self.global_indices)
        self.current_frame = 0
        self.playing = False

        self.data = {}
        for key in DATA_KEYS:
            f = ds._features.get(key)
            if f is not None:
                ts, vals = f.load(ep_idx)
                names = ds.meta.features.get(key, {}).get("names", None)
                self.data[key] = {"ts": ts, "vals": vals, "names": names}

        self._build_ui()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _build_ui(self):
        n_cams = len(CAMERA_KEYS)
        n_data = len(DATA_KEYS)

        data_rows = (n_data + 1) // 2
        total_rows = 1 + data_rows
        ratios = [1.8] + [1.0] * data_rows

        self.fig = plt.figure(figsize=(20, 3.5 + 3.0 * data_rows))
        gs = GridSpec(total_rows, n_cams + 2, figure=self.fig,
                      height_ratios=ratios, hspace=0.4, wspace=0.3,
                      left=0.03, right=0.97, top=0.94, bottom=0.05)

        # ── Top row: cameras ──
        self.cam_axes = []; self.cam_imgs = []
        for ci in range(n_cams):
            ax = self.fig.add_subplot(gs[0, ci])
            ax.axis("off")
            self.cam_axes.append(ax)
            self.cam_imgs.append(ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8)))

        # ── Status panel ──
        ax_status = self.fig.add_subplot(gs[0, n_cams:])
        ax_status.axis("off")
        ax_status.set_xlim(0, 1); ax_status.set_ylim(0, 1)
        self.status_text = ax_status.text(
            0.1, 0.95, "", fontsize=18, fontfamily="monospace",
            va="top", color="#4fc3f7", fontweight="bold",
        )
        self.status_detail = ax_status.text(
            0.1, 0.7, "", fontsize=12, fontfamily="monospace",
            va="top", color="#aaa",
        )
        # Progress bar background
        self.progress_bar = ax_status.add_patch(
            plt.Rectangle((0.1, 0.55), 0.8, 0.03, facecolor="#333", edgecolor="#555")
        )
        self.progress_fill = ax_status.add_patch(
            plt.Rectangle((0.1, 0.55), 0, 0.03, facecolor="#4fc3f7")
        )
        # Legend
        legend_lines = [
            "space=play  ←→=step  ↑↓=speed  w/s=window",
            f"episode {self.ep_idx}  |  {n_cams} cameras  |  {n_data} sensors",
        ]
        ax_status.text(0.1, 0.15, "\n".join(legend_lines),
                       fontsize=7, fontfamily="monospace", color="#666", va="bottom")

        # ── Data rows ──
        half = (n_cams + 2) // 2
        self.data_axes = {}; self.lines = {}
        for i, key in enumerate(DATA_KEYS):
            row = 1 + i // 2
            col_start = (i % 2) * half
            col_span = half if half > 0 else n_cams + 2
            ax = self.fig.add_subplot(gs[row, col_start:col_start + col_span])
            self.data_axes[key] = ax

            d = self.data[key]
            D = d["vals"].shape[1]
            names = d["names"]
            cmap = _color_map(key)
            line_objs = []
            for j in range(D):
                color = cmap(j / max(D - 1, 1)) if D > 1 else "#4fc3f7"
                l, = ax.plot([], [], lw=0.8, color=color)
                line_objs.append(l)
            self.lines[key] = line_objs

            short_key = key.split(".")[-1]
            ax.set_title(f"{short_key}  ({D}d, {len(d['ts'])} samples)", fontsize=13, color="#ccc")
            ax.set_xlabel("time (s)", fontsize=10, color="#666")
            ax.grid(True, alpha=0.2)
            ax.tick_params(colors="#666")
            if names and len(names) <= 8:
                ax.legend(line_objs, names, fontsize=8, loc="upper right",
                          ncol=min(4, len(names)), framealpha=0.3)
            # Cursor line
            self._cursor = ax.axvline(0, color="#ff5252", lw=0.8, alpha=0.7)

    def _draw(self):
        gidx = self.global_indices[self.current_frame]
        item = ds[int(gidx)]
        ts = item["timestamp"].item()

        # Cameras
        for ci, cam_key in enumerate(CAMERA_KEYS):
            frame = item[cam_key]
            if hasattr(frame, "numpy"): frame = frame.numpy()
            if frame.ndim == 3 and frame.shape[0] in (1, 3):
                frame = np.transpose(frame, (1, 2, 0))
            if frame.dtype == np.float32 or frame.max() <= 1.0:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            else:
                frame = frame.clip(0, 255).astype(np.uint8)
            self.cam_imgs[ci].set_data(frame)
            label = cam_key.split(".")[-1]
            self.cam_axes[ci].set_title(f"{label}", fontsize=12, color="#aaa")

        # Status
        self.status_text.set_text(
            f"t = {ts:.3f}s    frame {self.current_frame + 1}/{self.n_frames}"
        )
        pct = (self.current_frame + 1) / self.n_frames
        self.progress_fill.set_width(0.8 * pct)

        # Collect feature shapes
        shapes = []
        for key in DATA_KEYS:
            val = item.get(key)
            if val is not None:
                shapes.append(f"{key.split('.')[-1]}: {list(val.shape)}")
            elif f"{key}_timestamps" not in str(item.keys()):
                pass  # no data
        shape_str = "  |  ".join(shapes[:4])  # first 4
        if len(shapes) > 4:
            shape_str += "\n" + "  |  ".join(shapes[4:])

        self.status_detail.set_text(
            f"shapes: {shape_str}\n"
            f"playing: {'▶' if self.playing else '⏸'}  |  "
            f"speed: {self.play_fps:.0f}fps  |  "
            f"window: {self.window_size}f ({self.window_size / ds.fps:.1f}s)"
        )

        # Data lines
        for key in DATA_KEYS:
            d = self.data[key]; lines = self.lines[key]
            ft_ts = d["ts"]; ft_vals = d["vals"]
            f = ds._features.get(key)
            window = f.spec.get("window") if f else None
            ax = self.data_axes[key]

            if isinstance(window, (list, tuple)):
                t_start = ts - self.window_size / ds.fps
                mask = (ft_ts > t_start) & (ft_ts <= ts + 1e-6)
                plot_ts = ft_ts[mask]; plot_vals = ft_vals[mask]
                if len(plot_ts) > 0:
                    for j in range(min(ft_vals.shape[1], len(lines))):
                        lines[j].set_data(plot_ts, plot_vals[:, j])
                else:
                    for j in range(len(lines)): lines[j].set_data([], [])
                ax.set_xlim(ts - self.window_size / ds.fps - 0.05, ts + 0.05)
            else:
                win_start = max(0, self.current_frame - self.window_size + 1)
                idx = np.arange(win_start, min(self.current_frame + 1, len(ft_ts)))
                if len(idx) > 0:
                    for j in range(min(ft_vals.shape[1], len(lines))):
                        lines[j].set_data(ft_ts[idx], ft_vals[idx, j])
                ax.set_xlim(ts - self.window_size / ds.fps - 0.05, ts + 0.05)

            ax.relim(); ax.autoscale_view(scaley=True)

    def _on_key(self, event):
        if event.key == " ": self.playing = not self.playing
        elif event.key == "right": self.current_frame = min(self.current_frame + 1, self.n_frames - 1)
        elif event.key == "left": self.current_frame = max(self.current_frame - 1, 0)
        elif event.key == "up": self.play_fps = min(60, self.play_fps + 5); self.interval = 1000 / self.play_fps
        elif event.key == "down": self.play_fps = max(1, self.play_fps - 5); self.interval = 1000 / self.play_fps
        elif event.key == "w": self.window_size = min(300, self.window_size + 30)
        elif event.key == "s": self.window_size = max(10, self.window_size - 30)
        elif event.key in ("q", "escape"): plt.close(self.fig); return
        else: return
        self._draw(); self.fig.canvas.draw_idle()

    def _animate(self, _frame):
        if self.playing and self.current_frame < self.n_frames - 1:
            self.current_frame += 1
        self._draw()
        return []

    def run(self):
        self._draw()
        anim = FuncAnimation(self.fig, self._animate, interval=self.interval,
                             cache_frame_data=False, save_count=0)
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--fps", type=float, default=10)
    args = p.parse_args()
    print(f"Episode {args.episode}: {len(DATA_KEYS)} sensors, {len(CAMERA_KEYS)} cameras")
    Visualizer(args.episode, args.fps).run()


if __name__ == "__main__":
    main()
