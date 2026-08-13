"""Demonstrate window-based alignment: range window, interpolate, nearest, read-time override."""

import shutil
from pathlib import Path
import numpy as np
from mf_lerobot import MultiFrequencyLeRobotDataset

root = Path(__file__).parent.parent / "data" / "window_demo"
if root.exists():
    shutil.rmtree(root)

FPS = 30

ds = MultiFrequencyLeRobotDataset.create(
    repo_id="window_demo", fps=FPS,
    features={
        "observation.imu": {
            "dtype": "float32", "shape": (6,),
            "names": ["ax", "ay", "az", "gx", "gy", "gz"],
            "fps": 1000,
            "window": (-0.033, 0.0),      # ← range window
            "tolerance_s": 0.002,
        },
        "observation.finger_pose": {
            "dtype": "float32", "shape": (21,),
            "names": [f"j{i}" for i in range(21)],
            "fps": 120,
            "window": "interpolate",        # ← interpolate
            "tolerance_s": 0.01,
        },
        "observation.temperature": {
            "dtype": "float32", "shape": (1,), "names": ["temp"],
            "fps": 10,                       # ← nearest (no window)
            "tolerance_s": 0.05,
        },
    },
    root=root, use_videos=False,
)

# Record 1 episode
for i in range(60):
    t = i / FPS

    # IMU at 1000Hz
    for _ in range(33):
        ds.add_frame("observation.imu",
                     np.sin(t * 10 + np.arange(6) * 0.5).astype(np.float32))

    # Finger pose at 120Hz
    for _ in range(4):
        ds.add_frame("observation.finger_pose",
                     (np.sin(t * 5 + np.arange(21) * 0.1) * 0.5).astype(np.float32))

    # Temperature at 10Hz — once every 3 frames
    if i % 3 == 0:
        ds.add_frame("observation.temperature",
                     np.array([25.0 + i * 0.1], dtype=np.float32))

    ds.add_frame("task", f"step_{i}")
ds.save_episode()

print("=== Default windows ===")
item = ds[10]
ts = item["timestamp"].item()
print(f"\nFrame 10 (t={ts:.4f}):")
imu = item["observation.imu"]
print(f"  IMU [window=(-0.033,0)]: {imu.shape} readings, "
      f"range [{item['observation.imu_timestamps'][0].item():.4f}, "
      f"{item['observation.imu_timestamps'][-1].item():.4f}]")
fp = item["observation.finger_pose"]
print(f"  Finger [window=interpolate]: {fp.shape}")
temp = item["observation.temperature"]
print(f"  Temperature [nearest]: {temp.item():.1f}°C")

print("\n=== Read-time window override ===")
ds2 = MultiFrequencyLeRobotDataset(
    repo_id="window_demo", root=root,
    window_overrides={
        "observation.imu": (-0.066, 0.0),           # 2× wider
        "observation.finger_pose": (-0.033, 0.0),    # range instead of interpolate
    },
)
item2 = ds2[10]
imu2 = item2["observation.imu"]
print(f"  IMU [override=(-0.066,0)]: {imu2.shape} readings, "
      f"range [{item2['observation.imu_timestamps'][0].item():.4f}, "
      f"{item2['observation.imu_timestamps'][-1].item():.4f}]")
fp2 = item2["observation.finger_pose"]
print(f"  Finger [override to range]: {fp2.shape} readings, "
      f"timestamps: {item2['observation.finger_pose_timestamps'].shape}")
