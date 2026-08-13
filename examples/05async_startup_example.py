"""Demonstrate asynchronous sensor startup: master-clock alignment with per-sensor timestamp_start.

Scenario (wall clock)
---------------------
Each sensor comes online at a different time after the process starts, so its
first frame is returned at its own latency:

    observation.imu          1000 Hz   first frame at  0.04 s
    observation.emg           100 Hz   first frame at  0.12 s
    observation.state          30 Hz   first frame at  0.18 s  ← master clock reference
    observation.temperature    10 Hz   first frame at  0.22 s
    action                     30 Hz   first frame at  0.24 s
    observation.eeg           256 Hz   first frame at  0.31 s

Alignment procedure
-------------------
1. t_all_started   — the moment every sensor has delivered its first frame.
2. t = 0           — the master clock reference sensor's first frame at/after
                     t_all_started; its earlier frames are discarded.
3. timestamp_start — for every other sensor, the delta between its next frame
                     (first frame at/after t = 0) and the t = 0 frame; stored
                     in the feature spec so auto-generated timestamps stay
                     aligned with the master clock.
4. Record everything from t = 0 onward. The episode ends when the master clock
   stops, but every other sensor keeps streaming until its next frame after
   the master's last one — that trailing sample keeps the last master frame's
   window full.
"""

import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from mf_lerobot import MultiFrequencyLeRobotDataset
from mf_lerobot.utils import DEFAULT_DATA_PATH

FPS = 30
DURATION = 2.0                    # seconds recorded after t = 0
FRAMES = int(FPS * DURATION)      # master frames in the episode
MASTER = "observation.state"      # master clock reference sensor

# key: (fps, first-frame latency [s], per-frame jitter σ [s], alignment tolerance_s)
# The checker compares every master frame with the sensor's NEAREST sample, so
# the worst case is half the sensor's period — tolerance must cover that plus
# jitter (e.g. 10 Hz → ≤ 50 ms away → tolerance 0.06).
SENSORS = {
    "observation.imu":         (1000, 0.04, 0.0002, 0.005),
    "observation.emg":         (100,  0.12, 0.001,  0.01),
    "observation.state":       (30,   0.18, 0.0005, 0.001),
    "observation.temperature": (10,   0.22, 0.001,  0.06),
    "action":                  (30,   0.24, 0.0005, 0.03),
    "observation.eeg":         (256,  0.31, 0.0005, 0.005),
}

rng = np.random.default_rng(42)


# ── 1. Simulate raw wall-clock streams ──────────────────────────────────────
# Each sensor returns frames from its own latency onward, with small jitter,
# far enough past the planned stop so the sync step has tail frames to pick.

horizon = max(lat for _, lat, _, _ in SENSORS.values()) + DURATION + 1.0
streams = {}
for key, (fps, latency, jitter, _) in SENSORS.items():
    n = int(np.ceil((horizon - latency) * fps))
    ts = latency + np.arange(n) / fps + rng.normal(0.0, jitter, n)
    streams[key] = np.sort(ts)


# ── 2. Align: master t=0, per-sensor timestamp_start, per-sensor cutoff ─────

# 2a. When has EVERY sensor delivered its first frame?
t_all_started = max(ts[0] for ts in streams.values())

# 2b. The master clock reference sensor's first frame at/after that moment
#     is t = 0; its earlier frames are discarded.
master_wall = streams[MASTER][streams[MASTER] >= t_all_started][:FRAMES]
t0 = master_wall[0]
t_master_stop = master_wall[-1]          # episode ends when the master stops

# 2c. For every other sensor: timestamp_start = next frame (at/after t = 0)
#     minus the t = 0 frame; keep its samples from there until its next frame
#     after the master clock stopped.
timestamp_start, keep = {}, {}
for key, ts in streams.items():
    if key == MASTER:
        continue
    usable = ts[ts >= t0]                 # frames from t = 0 onward
    timestamp_start[key] = usable[0] - t0
    tail = usable[usable > t_master_stop]  # next frame after the master stopped
    cutoff = tail[0] if len(tail) else usable[-1]
    keep[key] = usable[usable <= cutoff]


# ── Synthetic data generators ───────────────────────────────────────────────

def _state(t):
    return np.array([
        0.5 + 0.15 * np.sin(t * 0.8),
        0.10 * np.cos(t * 0.6),
        0.3 + 0.08 * np.sin(t * 1.0),
        0.2 * np.sin(t * 1.2),
        0.15 * np.cos(t * 0.9),
        0.1 * np.sin(t * 1.5),
        0.5 + 0.5 * np.sin(t * 0.7),
    ], dtype=np.float32)


def _action(t):
    return np.array([
        0.12 * np.cos(t * 0.8), 0.06 * np.sin(t * 0.6), 0.08 * np.cos(t * 1.0),
        0.24 * np.cos(t * 1.2), 0.14 * np.sin(t * 0.9), 0.15 * np.cos(t * 1.5),
        0.35 * np.cos(t * 0.7),
    ], dtype=np.float32)


def _imu(t):
    return np.array([
        0.3 * np.sin(t * 3.0) + rng.normal(0.0, 0.02),
        0.2 * np.cos(t * 2.5) + rng.normal(0.0, 0.02),
        -9.81 + 0.5 * np.sin(t * 1.5) + rng.normal(0.0, 0.03),
        0.1 * np.sin(t * 4.0) + rng.normal(0.0, 0.005),
        0.08 * np.cos(t * 3.5) + rng.normal(0.0, 0.005),
        0.05 * np.sin(t * 2.0) + rng.normal(0.0, 0.003),
    ], dtype=np.float32)


def _eeg(t):
    return np.array([
        5.0 * np.sin(2 * np.pi * 10 * t + c * 0.8) +
        2.0 * np.sin(2 * np.pi * 20 * t + c * 1.2) +
        rng.normal(0.0, 1.5)
        for c in range(8)
    ], dtype=np.float32)


def _emg(t):
    return np.array([
        (0.3 + 0.2 * c) * np.abs(np.sin(t * 15 + c * 2.0)) + rng.normal(0.0, 0.02)
        for c in range(4)
    ], dtype=np.float32)


def _temp(t):
    return np.array([25.0 + 2.0 * np.sin(t * 0.3) + rng.normal(0.0, 0.05)],
                    dtype=np.float32)


GENERATORS = {
    "observation.imu": _imu,
    "observation.emg": _emg,
    "observation.eeg": _eeg,
    "observation.temperature": _temp,
    "action": _action,
}


# ── Create dataset with the computed timestamp_start values ─────────────────

root = Path(__file__).parent.parent / "data" / "async_startup_demo"
if root.exists():
    shutil.rmtree(root)

features = {
    "observation.imu": {
        "dtype": "float32", "shape": (6,),
        "names": ["ax", "ay", "az", "gx", "gy", "gz"],
        "fps": SENSORS["observation.imu"][0], "window": (-0.033, 0.0),
        "timestamp_start": float(timestamp_start["observation.imu"]),
        "tolerance_s": SENSORS["observation.imu"][3],
    },
    "observation.emg": {
        "dtype": "float32", "shape": (4,),
        "names": ["flex_carpi", "ext_carpi", "biceps", "triceps"],
        "fps": SENSORS["observation.emg"][0], "window": (-0.033, 0.0),
        "timestamp_start": float(timestamp_start["observation.emg"]),
        "tolerance_s": SENSORS["observation.emg"][3],
    },
    "observation.state": {
        "dtype": "float32", "shape": (7,),
        "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        "fps": SENSORS["observation.state"][0],
        "tolerance_s": SENSORS["observation.state"][3],
    },
    "action": {
        "dtype": "float32", "shape": (7,),
        "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "dgrip"],
        "fps": SENSORS["action"][0],
        "timestamp_start": float(timestamp_start["action"]),
        "tolerance_s": SENSORS["action"][3],
    },
    "observation.eeg": {
        "dtype": "float32", "shape": (8,),
        "names": ["Fz", "Cz", "Pz", "Oz", "F3", "F4", "C3", "C4"],
        "fps": SENSORS["observation.eeg"][0], "window": (-0.033, 0.0),
        "timestamp_start": float(timestamp_start["observation.eeg"]),
        "tolerance_s": SENSORS["observation.eeg"][3],
    },
    "observation.temperature": {
        "dtype": "float32", "shape": (1,), "names": ["temp"],
        "fps": SENSORS["observation.temperature"][0],
        "timestamp_start": float(timestamp_start["observation.temperature"]),
        "tolerance_s": SENSORS["observation.temperature"][3],
    },
}

ds = MultiFrequencyLeRobotDataset.create(
    repo_id="async_startup_demo", fps=FPS, features=features,
    root=root, use_videos=False,
)


# ── 3. Record: master clock first, then every other sensor's stream ─────────

# Master clock reference sensor → one task record + one state sample per frame
# (task calls carry the master timestamps into master_index.parquet).
for i, tw in enumerate(master_wall):
    ds.add_frame("task", f"step_{i}", timestamp=float(tw - t0))
    ds.add_frame(MASTER, _state(tw), timestamp=float(tw - t0))

# Every other sensor: samples from its first frame at/after t = 0 through its
# next frame after the master stopped; timestamps are wall clock minus t0.
for key, ts_wall in keep.items():
    gen = GENERATORS[key]
    for tw in ts_wall:
        ds.add_frame(key, gen(tw), timestamp=float(tw - t0))

ds.save_episode()


# ── Report the alignment ────────────────────────────────────────────────────

print("\n=== Sensor startup (wall clock) ===")
for key, (fps, latency, jitter, _) in SENSORS.items():
    print(f"  {key:24s} {fps:>5d} Hz  first frame at {latency:5.2f} s "
          f"(jitter σ={jitter:.4f} s)")

print("\n=== Alignment ===")
master_start_frame = int(np.searchsorted(streams[MASTER], t0))
print(f"  all sensors started at wall t = {t_all_started:.4f} s")
print(f"  master t=0 at wall t          = {t0:.4f} s "
      f"({MASTER} frame #{master_start_frame}; the {master_start_frame} "
      f"earlier frames are discarded)")
print(f"  master stops at wall t        = {t_master_stop:.4f} s "
      f"(episode t = {t_master_stop - t0:.4f} s, {FRAMES} frames)")

print("\n=== Per-sensor timestamp_start / cutoff ===")
print(f"  {'feature':24s} {'timestamp_start':>15s} {'samples':>8s} "
      f"{'first@':>9s} {'last@':>9s} {'tail':>5s}")
for key in keep:
    first_rel = keep[key][0] - t0
    last_rel = keep[key][-1] - t0
    n_tail = int(np.sum(keep[key] > t_master_stop))
    print(f"  {key:24s} {timestamp_start[key]:>15.4f} {len(keep[key]):>8d} "
          f"{first_rel:>9.4f} {last_rel:>9.4f} {n_tail:>5d}")

print("\n=== Verify what was written to disk ===")
for key in keep:
    fpath = root / DEFAULT_DATA_PATH.format(
        episode_chunk=0, episode_index=0, feature_key=key)
    ts = pq.read_table(fpath).column("timestamp").to_numpy()
    assert np.isclose(ts[0], timestamp_start[key], atol=1e-6), key
    assert ts[-1] > t_master_stop - t0, key
    print(f"  {key:24s} stored [{ts[0]:.4f}, {ts[-1]:.4f}] s "
          f"({len(ts)} rows) — first == timestamp_start, "
          f"last past master stop ✓")


# ── 4. Read back ────────────────────────────────────────────────────────────

ds2 = MultiFrequencyLeRobotDataset(repo_id="async_startup_demo", root=root)

item0 = ds2[0]
print("\n=== Read back ===")
print(f"frame 0 (t = {item0['timestamp'].item():.4f}):")
print(f"  state:            {item0['observation.state'][:3].tolist()} ...")
print(f"  imu window:       {item0['observation.imu'].shape[0]} readings "
      f"(empty — every sensor's first stored frame is at its timestamp_start > 0)")
print(f"  temperature:      {item0['observation.temperature'].item():.2f}°C (nearest)")

mid = ds2[FRAMES // 2]
print(f"\nframe {FRAMES // 2} (t = {mid['timestamp'].item():.4f}):")
print(f"  imu window:       {mid['observation.imu'].shape[0]} readings")
print(f"  eeg window:       {mid['observation.eeg'].shape[0]} readings")

last = ds2[FRAMES - 1]
print(f"\nframe {FRAMES - 1} (t = {last['timestamp'].item():.4f}, "
      f"master just stopped):")
print(f"  imu window:       {last['observation.imu'].shape[0]} readings "
      f"(still full — the sensor kept streaming one frame past the master stop)")
