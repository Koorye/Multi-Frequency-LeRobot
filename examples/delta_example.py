"""Demonstrate delta_timestamps: multi-frame temporal window queries."""

import shutil
from pathlib import Path
import numpy as np
from mf_lerobot import MultiFrequencyLeRobotDataset

root = Path(__file__).parent.parent / "data" / "delta_demo"
if root.exists():
    shutil.rmtree(root)

FPS = 30

ds = MultiFrequencyLeRobotDataset.create(
    repo_id="delta_demo", fps=FPS,
    features={
        "observation.state": {
            "dtype": "float32", "shape": (3,),
            "names": ["x", "y", "z"],
            "fps": 30, "tolerance_s": 0.001,
        },
        "observation.imu": {
            "dtype": "float32", "shape": (6,),
            "names": ["ax", "ay", "az", "gx", "gy", "gz"],
            "fps": 1000, "window": (-0.033, 0.0),
        },
    },
    root=root, use_videos=False,
)

# Record linear motion
for i in range(60):
    t = i / FPS
    for _ in range(33):
        ds.add_frame("observation.imu", np.random.randn(6).astype(np.float32) * 0.1)
    ds.add_frame("observation.state", np.array([t, t * 2, t * 3], dtype=np.float32))
    ds.add_frame("task", f"step_{i}")
ds.save_episode()

# Load with delta_timestamps: query state at t-0.1, t, t+0.1
ds2 = MultiFrequencyLeRobotDataset(
    repo_id="delta_demo", root=root,
    delta_timestamps={"observation.state": [-0.1, 0.0, 0.1]},
)

item = ds2[20]
ts = item["timestamp"].item()
print(f"\nFrame 20 (t={ts:.4f}):")
print(f"  state shape: {item['observation.state'].shape}")  # [3, 3] — 3 timepoints × 3 dims
print(f"  state values:\n{item['observation.state']}")

# Without delta: single timepoint
item_single = ds[20]
print(f"\nWithout delta_timestamps:")
print(f"  state shape: {item_single['observation.state'].shape}")  # [3]
print(f"  state: {item_single['observation.state']}")
