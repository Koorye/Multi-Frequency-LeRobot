#!/usr/bin/env python
"""Convert Embodied-Brain-Collect ``session-night`` recordings into an mf_lerobot dataset.

Usage::

    python scripts/convert_session_night.py                  # 当天
    python scripts/convert_session_night.py 2026-08-24      # 指定日期
    python scripts/convert_session_night.py 2026-08-24 --out data/raw/night

All session directories matching ``<date>-*`` under the source dir are packed
into ONE dataset — each session directory becomes one episode.

Mapping (session stream -> dataset feature):

============================  ==============================  =====================
source                       feature                         notes
============================  ==============================  =====================
cam_head                     observation.images.head_rgb     master clock, 30 fps
cam_left_wrist               observation.images.left_wrist_rgb
cam_right_wrist              observation.images.right_wrist_rgb
cam_third                    observation.images.third_rgb
eye/frames (scene video)     observation.images.eye_scene_rgb
eeg                          observation.eeg                 132 channels, 1000 Hz
emg_left                     observation.emg_left            8 channels, ~2000 Hz
emg_right                    observation.emg_right           8 channels, ~2000 Hz
emg_left (imu)               observation.imu_left            6d accel+gyro
emg_right (imu)              observation.imu_right           6d accel+gyro
eye (imu)                    observation.eye_imu             6d accel+gyro
eye (gaze)                   observation.gaze                2d (x, y)
hand_pose (ergonomics)       observation.hand_pose           40d finger joint angles
hand_pose (skeleton pos)     observation.hand_skeleton_pos   50 nodes x 3d
hand_pose (skeleton rot)     observation.hand_skeleton_rot   50 nodes x 4d quaternion
position                     observation.state               3 trackers x (x,y,z,rpy,valid)
marker                       (episode window + task labels)  RUN_START..RUN_END
============================  ==============================  =====================

By default each episode covers the ``RUN_START``..``RUN_END`` marker window
(same as ``qc_report.json``); pass ``--full`` to keep the whole recording.
All timestamps are rebased to the episode's first master frame, matching the
LeRobot convention (float32-safe, per-episode relative time).
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mf_lerobot import MultiFrequencyLeRobotDataset  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path("/home/baai/Projects/Embodied-Brain-Collect/data/session-night")

MASTER_FPS = 30.0
WINDOW = (-0.033, 0.0)  # (-33 ms, 0 ms] — same as the demo datasets

# ── Stream → feature mapping ──────────────────────────────────────────────

# dir name -> (feature key, npz timestamp key, mp4 filename)
VIDEO_STREAMS: dict[str, tuple[str, str, str]] = {
    "cam_head": ("observation.images.head_rgb", "frames_timestamps", "frames.mp4"),
    "cam_left_wrist": ("observation.images.left_wrist_rgb", "frames_timestamps", "frames.mp4"),
    "cam_right_wrist": ("observation.images.right_wrist_rgb", "frames_timestamps", "frames.mp4"),
    "cam_third": ("observation.images.third_rgb", "frames_timestamps", "frames.mp4"),
    "eye": ("observation.images.eye_scene_rgb", "scene_timestamps", "eye.mp4"),
}

# feature keys that get a sliding query window at read time
WINDOWED_FEATURES = {
    "observation.eeg",
    "observation.emg_left",
    "observation.emg_right",
    "observation.imu_left",
    "observation.imu_right",
    "observation.eye_imu",
    "observation.gaze",
    "observation.hand_pose",
    "observation.hand_skeleton_pos",
    "observation.hand_skeleton_rot",
}

IMU_NAMES = ["ax", "ay", "az", "gx", "gy", "gz"]


# ── Loaders ───────────────────────────────────────────────────────────────

def _spread_run_timestamps(ts: np.ndarray) -> np.ndarray:
    """Spread read-arrival timestamps across each run of identical values.

    The Weili armband streams EMG/IMU frames continuously; the recorder
    stamps every frame carried by one ``Serial.read()`` with that read's PC
    clock, so runs of frames share a timestamp (see ``weili_emg_recorder``
    docstring).  Samples within a run are assumed uniformly spaced between
    the previous and the current read, which recovers per-sample timing
    while keeping the order implied by the wire sequence numbers.
    """
    n = len(ts)
    out = np.empty_like(ts)
    dt_global = (ts[-1] - ts[0]) / max(n - 1, 1) if n > 1 else 0.0
    prev_t = None
    i = 0
    while i < n:
        j = i + 1
        while j < n and ts[j] == ts[i]:
            j += 1
        k = j - i
        if prev_t is None:
            t0 = ts[i] - k * dt_global  # first run: extend backward at global rate
        else:
            t0 = prev_t
        out[i:j] = t0 + (np.arange(k) + 1) * (ts[i] - t0) / k
        prev_t = ts[i]
        i = j
    return out


def _load_imu_stream(z, ts_key):
    """(accel, gyro) -> (N, 6) float32 with IMU_NAMES."""
    return (
        _spread_run_timestamps(z[ts_key].astype(np.float64)),
        np.hstack([z["imu_accel"], z["imu_gyro"]]).astype(np.float32),
        IMU_NAMES,
    )


def load_parquet_streams(session_dir: Path) -> list[tuple[str, np.ndarray, np.ndarray, list[str] | None]]:
    """Load all non-video streams of one session.

    Returns list of (feature_key, timestamps_abs_pc, values_2d, names|None).
    """
    out: list[tuple[str, np.ndarray, np.ndarray, list[str] | None]] = []

    # EEG — 132 channels (last column of the npz is the Trigger channel)
    p = session_dir / "eeg" / "eeg.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        if "eeg_timestamps_pc" in z:
            ts = z["eeg_timestamps_pc"].astype(np.float64)
        elif bool(z["eeg_fit_fitted"]) and "eeg_fit_slope_pc_per_eeg" in z:
            # alignment ran but pc timestamps were not exported — rebuild from the fit
            rate = float(z["eeg_sample_rate"])
            ts = (float(z["eeg_fit_pc_t0_s"])
                  + np.arange(len(z["eeg_data"])) * float(z["eeg_fit_slope_pc_per_eeg"]) / rate)
            print(f"[eeg] '{session_dir.name}': rebuilding timestamps from fit")
        else:
            print(f"[eeg] '{session_dir.name}': no pc-aligned timestamps — skipping eeg")
            ts = None
        if ts is not None:
            n_ch = int(z["eeg_n_eeg_channels"])
            names = [str(s) for s in z["eeg_channel_names"][:n_ch]]
            out.append((
                "observation.eeg",
                ts,
                z["eeg_data"][:, :n_ch].astype(np.float32),
                names,
            ))

    # EMG armbands — 8 channels each + on-board IMU
    for side in ("left", "right"):
        p = session_dir / f"emg_{side}" / f"emg_{side}.npz"
        if p.exists():
            z = np.load(p, allow_pickle=True)
            out.append((
                f"observation.emg_{side}",
                _spread_run_timestamps(z["emg_timestamps"].astype(np.float64)),
                z["emg_data"].astype(np.float32),
                None,
            ))
            if "imu_timestamps" in z:
                out.append((f"observation.imu_{side}",) + _load_imu_stream(z, "imu_timestamps"))

    # Eye tracker — gaze + IMU (scene video handled in VIDEO_STREAMS)
    p = session_dir / "eye" / "eye.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        if "gaze_timestamps" in z:
            out.append((
                "observation.gaze",
                z["gaze_timestamps"].astype(np.float64),
                z["gaze_xy"].astype(np.float32),
                ["x", "y"],
            ))
        if "imu_timestamps" in z:
            out.append(("observation.eye_imu",) + _load_imu_stream(z, "imu_timestamps"))

    # Hand pose — 40d ergonomics + skeleton (50 nodes)
    p = session_dir / "hand_pose" / "hand_pose.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        if "ergo_timestamps" in z and "ergo_data" in z:
            out.append((
                "observation.hand_pose",
                z["ergo_timestamps"].astype(np.float64),
                z["ergo_data"].astype(np.float32),
                None,
            ))
        if "skeleton_timestamps" in z and "skeleton_positions" in z:
            out.append((
                "observation.hand_skeleton_pos",
                z["skeleton_timestamps"].astype(np.float64),
                z["skeleton_positions"].reshape(-1, 50 * 3).astype(np.float32),
                None,
            ))
        if "skeleton_timestamps" in z and "skeleton_rotations" in z:
            out.append((
                "observation.hand_skeleton_rot",
                z["skeleton_timestamps"].astype(np.float64),
                z["skeleton_rotations"].reshape(-1, 50 * 4).astype(np.float32),
                None,
            ))

    # Position trackers — 3 devices x (x, y, z, roll, pitch, yaw, valid)
    p = session_dir / "position" / "position.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        n_dev = len(z["valid"][0])
        cols, names = [], []
        for i in range(n_dev):
            cols += [
                z["positions_m"][:, i, :],
                z["euler_rpy_deg"][:, i, :],
                z["valid"][:, i:i + 1].astype(np.float32),
            ]
            names += [
                f"dev{i}_x", f"dev{i}_y", f"dev{i}_z",
                f"dev{i}_roll_deg", f"dev{i}_pitch_deg", f"dev{i}_yaw_deg",
                f"dev{i}_valid",
            ]
        out.append((
            "observation.state",
            z["timestamps_s"].astype(np.float64),
            np.concatenate(cols, axis=1).astype(np.float32),
            names,
        ))

    return out


def load_marker_window(session_dir: Path) -> tuple[float | None, float | None]:
    """(RUN_START, RUN_END) pc timestamps, or (None, None) without markers."""
    p = session_dir / "marker" / "marker.npz"
    if not p.exists():
        return None, None
    z = np.load(p, allow_pickle=True)
    t0 = t1 = None
    for tag, t in zip(z["marker_tag"], z["marker_t_sent_pc"]):
        if tag == "RUN_START":
            t0 = float(t)
        elif tag == "RUN_END":
            t1 = float(t)
    return t0, t1


def load_task_name(session_dir: Path) -> str:
    p = session_dir / "meta.yaml"
    if p.exists():
        meta = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if meta.get("task_name"):
            return str(meta["task_name"])
    return f"session_{session_dir.name}"


def _window_mask(ts: np.ndarray, win: tuple[float | None, float | None]) -> np.ndarray:
    mask = np.ones(len(ts), dtype=bool)
    if win[0] is not None:
        mask &= ts >= win[0]
    if win[1] is not None:
        mask &= ts <= win[1]
    return mask


def load_master_frames(session_dir: Path, win: tuple[float | None, float | None]) -> np.ndarray | None:
    """Absolute pc timestamps of the master frames (cam_head, falling back to any camera)."""
    order = ["cam_head"] + [d for d in VIDEO_STREAMS if d != "cam_head"]
    for stream_dir in order:
        npz = session_dir / stream_dir / f"{stream_dir}.npz"
        if not npz.exists():
            continue
        ts = np.load(npz, allow_pickle=True)[VIDEO_STREAMS[stream_dir][1]].astype(np.float64)
        ts = ts[_window_mask(ts, win)]
        if len(ts):
            if stream_dir != "cam_head":
                print(f"[master] '{session_dir.name}': cam_head missing, using '{stream_dir}'")
            return ts
    return None


# ── Feature spec ──────────────────────────────────────────────────────────

def _stream_rate(ts: np.ndarray) -> float:
    """Sample rate = n / duration (robust to duplicate read-arrival timestamps)."""
    if len(ts) < 2:
        return 0.0
    duration = ts[-1] - ts[0]
    return round(len(ts) / duration, 1) if duration > 0 else 0.0


def probe_video_shape(mp4: Path) -> tuple[int, int, int]:
    import av
    with av.open(str(mp4)) as container:
        frame = next(container.decode(video=0))
        h, w = frame.height, frame.width
        return h, w, 3


def build_feature_specs(sessions: list[Path]) -> dict[str, dict]:
    """Union of all streams present across the matched sessions.

    Shape/names/rate are probed from the first session that has the stream.
    """
    specs: dict[str, dict] = {}

    def _first_session_with(stream_dir: str) -> Path | None:
        for sd in sessions:
            if (sd / stream_dir).is_dir():
                return sd
        return None

    # Videos — dict order decides info.json's master_feature (head first)
    for stream_dir, (key, ts_key, mp4_name) in VIDEO_STREAMS.items():
        npz_name = "eye.npz" if stream_dir == "eye" else f"{stream_dir}.npz"
        sd = _first_session_with(stream_dir)
        if sd is None:
            continue
        mp4 = sd / stream_dir / mp4_name
        npz = sd / stream_dir / npz_name
        if not mp4.exists() or not npz.exists():
            print(f"[spec] '{stream_dir}' present but npz/mp4 missing — skipping feature")
            continue
        try:
            h, w, c = probe_video_shape(mp4)
        except Exception as e:
            print(f"[spec] '{stream_dir}': cannot decode {mp4.name} ({e}) — skipping feature")
            continue
        specs[key] = {
            "dtype": "video",
            "shape": (h, w, c),
            "names": ["h", "w", "c"],
            # int: lerobot's encode_video_frames passes fps straight to av,
            # which rejects floats ("'float' object has no attribute 'numerator'")
            "fps": int(MASTER_FPS),
            "tolerance_s": 0.001,
        }

    # Parquet streams — union over sessions (probed from the first one that has it)
    for sd in sessions:
        for key, ts, vals, names in load_parquet_streams(sd):
            if key in specs or not len(ts):
                continue
            rate = _stream_rate(ts)
            specs[key] = {
                "dtype": "float32",
                "shape": (vals.shape[1],),
                "names": names,
                "fps": rate,
                "tolerance_s": max(0.005, 3.0 / max(rate, 1.0)),
            }
            if key in WINDOWED_FEATURES:
                specs[key]["window"] = WINDOW

    return specs


# ── Episode writer ────────────────────────────────────────────────────────

def write_video_frames(vf, mp4: Path, cam_ts: np.ndarray,
                       master_abs: np.ndarray, master_rel: np.ndarray) -> bool:
    """Feed one camera: pick the frame nearest to each master timestamp.

    The source mp4 may hold more frames than timestamps (recorder shutdown
    race) — extra frames are ignored; missing ones are filled with the last
    decoded frame so the video keeps exactly ``len(master_abs)`` frames.
    Returns False if the mp4 yields no frames at all.
    """
    import av

    n_ts = len(cam_ts)
    idx = np.searchsorted(cam_ts, master_abs)
    idx = np.clip(idx, 0, n_ts - 1)
    prev = np.clip(idx - 1, 0, n_ts - 1)
    use_prev = np.abs(cam_ts[prev] - master_abs) < np.abs(cam_ts[idx] - master_abs)
    sel = np.where(use_prev, prev, idx)

    need: dict[int, list[int]] = defaultdict(list)
    for j, i in enumerate(sel.tolist()):
        need[i].append(j)

    written = np.zeros(len(master_abs), dtype=bool)
    last = None
    with av.open(str(mp4)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i >= n_ts:
                break
            img = frame.to_ndarray(format="rgb24")
            last = img
            for j in need.get(i, ()):
                vf.add(img, float(master_rel[j]))
                written[j] = True

    if last is None:
        return False
    miss = int((~written).sum())
    if miss:
        print(f"[video] '{vf.key}': {miss} master frames without a decoded camera frame"
              f" — filling with last frame")
        for j in np.where(~written)[0]:
            vf.add(last, float(master_rel[j]))
    return True


def write_episode(ds, session_dir: Path, task_name: str, master_abs: np.ndarray,
                  streams: list[tuple[str, np.ndarray, np.ndarray, list[str] | None]],
                  win: tuple[float | None, float | None]) -> None:
    t0 = float(master_abs[0])
    master_rel = (master_abs - t0).astype(np.float64)

    # Master clock frames with the session's task label
    for t in master_rel:
        ds.add_frame("task", task_name, float(t))

    # Parquet streams — window-filtered to the marker window, rebased to t0
    present: set[str] = set()
    for key, ts, vals, _names in streams:
        mask = _window_mask(ts, win)
        ts_w, vals_w = ts[mask], vals[mask]
        if not len(ts_w):
            continue
        rel = (ts_w - t0).astype(np.float64)
        for t, v in zip(rel, vals_w):
            ds.add_frame(key, v, float(t))
        present.add(key)

    # Videos — resampled onto the master clock
    for stream_dir, (key, ts_key, mp4_name) in VIDEO_STREAMS.items():
        npz = session_dir / stream_dir / (f"{stream_dir}.npz" if stream_dir != "eye" else "eye.npz")
        mp4 = session_dir / stream_dir / mp4_name
        if not npz.exists() or not mp4.exists():
            continue
        cam_ts = np.load(npz, allow_pickle=True)[ts_key].astype(np.float64)
        if not len(cam_ts):
            continue
        if write_video_frames(ds._features[key], mp4, cam_ts, master_abs, master_rel):
            present.add(key)

    # Drop features this session doesn't have, then advance their episode state
    dropped = {k: ds._features.pop(k) for k in list(ds._features) if k not in present}
    ds.save_episode()
    for k, f in dropped.items():
        f.next_episode()
        ds._features[k] = f

    if dropped:
        print(f"[episode] '{session_dir.name}': skipped features "
              f"{sorted(dropped)} (absent in this session)")


def _cleanup_images(root: Path, ep_idx: int, feature_keys: list[str]) -> None:
    for key in feature_keys:
        d = root / "images" / key / f"episode_{ep_idx:06d}"
        if d.is_dir():
            shutil.rmtree(d)
        # drop the now-empty feature dir (rmdir ignores non-empty ones)
        try:
            d.parent.rmdir()
        except OSError:
            pass
    try:
        (root / "images").rmdir()
    except OSError:
        pass


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python scripts/convert_session_night.py 2026-08-24 --force",
    )
    p.add_argument("date", nargs="?", default=None,
                   help="日期 YYYY-MM-DD，默认当天 (default: %(default)s)")
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                   help=f"session-night 源目录 (default: {DEFAULT_SOURCE})")
    p.add_argument("--out", type=Path, default=None,
                   help="输出数据集目录 (default: data/raw/session-night-<date>)")
    p.add_argument("--force", action="store_true",
                   help="输出目录已存在时先删除")
    p.add_argument("--full", action="store_true",
                   help="使用整段录制，忽略 RUN_START..RUN_END 标记窗口")
    p.add_argument("--keep-images", action="store_true",
                   help="保留中间 PNG 帧（默认编码后删除）")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="只转换前 N 个会话（调试用）")
    args = p.parse_args(argv)

    if args.date is None:
        args.date = dt.date.today().isoformat()
    try:
        dt.date.fromisoformat(args.date)
    except ValueError:
        p.error(f"无效日期: {args.date!r} (应为 YYYY-MM-DD)")
    if args.out is None:
        args.out = PROJECT_ROOT / "data" / "raw" / f"session-night-{args.date}"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    sessions = sorted(p for p in (args.source / "").glob(f"{args.date}-*")
                      if p.is_dir() and p.name.startswith(f"{args.date}-"))
    if not sessions:
        print(f"[error] '{args.source}' 下没有匹配 {args.date}-* 的会话目录")
        return 1
    if args.max_episodes:
        sessions = sessions[: args.max_episodes]
    print(f"[input] {len(sessions)} 个会话: "
          + ", ".join(s.name for s in sessions[:5])
          + (" ..." if len(sessions) > 5 else ""))

    if args.out.exists():
        if not args.force:
            print(f"[error] 输出目录已存在: {args.out} (用 --force 覆盖)")
            return 1
        shutil.rmtree(args.out)
    if not args.full:
        missing_win = [s.name for s in sessions if load_marker_window(s) == (None, None)]
        if missing_win:
            print(f"[warn] 以下会话没有 marker 窗口，将使用整段录制: {missing_win}")

    specs = build_feature_specs(sessions)
    if not specs:
        print("[error] 所有会话都没有可用数据流")
        return 1
    print(f"[spec] {len(specs)} 个特征: " + ", ".join(specs))

    ds = MultiFrequencyLeRobotDataset.create(
        repo_id=args.out.name, fps=MASTER_FPS, features=specs,
        root=args.out, use_videos=True,
    )

    video_keys = [k for k, ft in specs.items() if ft.get("dtype") == "video"]
    for i, session_dir in enumerate(sessions):
        win = load_marker_window(session_dir) if not args.full else (None, None)
        master_abs = load_master_frames(session_dir, win)
        if master_abs is None or not len(master_abs):
            print(f"[warn] 跳过 '{session_dir.name}': 没有 master 帧")
            continue
        task_name = load_task_name(session_dir)
        streams = load_parquet_streams(session_dir)
        write_episode(ds, session_dir, task_name, master_abs, streams, win)
        if not args.keep_images:
            _cleanup_images(args.out, i, video_keys)
        print(f"[episode {i}] '{session_dir.name}' task='{task_name}' "
              f"{len(master_abs)} frames")

    print(f"[done] {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
