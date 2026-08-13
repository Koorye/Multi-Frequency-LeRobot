"""Demonstrate multi-episode recording and episode-filtered loading."""

import shutil
from pathlib import Path
import numpy as np
from mf_lerobot import MultiFrequencyLeRobotDataset

root = Path(__file__).parent.parent / "data" / "multi_ep_demo"
if root.exists():
    shutil.rmtree(root)

FPS = 30

ds = MultiFrequencyLeRobotDataset.create(
    repo_id="multi_ep_demo", fps=FPS,
    features={
        "observation.state": {
            "dtype": "float32", "shape": (3,),
            "names": ["x", "y", "z"],
            "fps": 30, "tolerance_s": 0.001,
        },
    },
    root=root, use_videos=False,
)

# Record 3 episodes with different motion patterns
TASKS = ["linear_motion", "circular_motion", "zigzag_motion"]
for ep, task in enumerate(TASKS):
    offset = ep * 2.0 * np.pi
    for i in range(30):  # 1 second each
        t = i / FPS
        if task == "linear_motion":
            s = np.array([t, t * 0.5, t * 2], dtype=np.float32)
        elif task == "circular_motion":
            s = np.array([np.sin(t * 3 + offset), np.cos(t * 3 + offset), 0.0], dtype=np.float32)
        else:
            s = np.array([t % 0.3, (t * 2) % 0.5, np.sin(t * 8 + offset)], dtype=np.float32)
        ds.add_frame("observation.state", s)
        ds.add_frame("task", f"{task}_{i}")
    ds.save_episode()
    print(f"Episode {ep} '{task}': {ds.num_frames} total frames")

# Load only episodes 0 and 2
print(f"\nFull dataset: {ds.num_frames} frames across {ds.meta.total_episodes} episodes")

ds_subset = MultiFrequencyLeRobotDataset(
    repo_id="multi_ep_demo", root=root, episodes=[0, 2],
)
print(f"Filtered [0, 2]: {ds_subset.num_frames} frames")

# Verify
for i in [0, 15, 29]:
    item = ds_subset[i]
    print(f"  frame {i}: ep={item['episode_index'].item()}, "
          f"ts={item['timestamp'].item():.3f}, task={item['task']}")
