"""Demonstrate per-feature tolerance check and alignment_report.jsonl."""

import json
import shutil
from pathlib import Path
import numpy as np
from mf_lerobot import MultiFrequencyLeRobotDataset

root = Path(__file__).parent.parent / "data" / "tolerance_demo"
if root.exists():
    shutil.rmtree(root)

FPS = 30

ds = MultiFrequencyLeRobotDataset.create(
    repo_id="tolerance_demo", fps=FPS,
    features={
        "observation.imu": {
            "dtype": "float32", "shape": (6,),
            "names": ["ax", "ay", "az", "gx", "gy", "gz"],
            "fps": 1000, "window": (-0.033, 0.0),
            "timestamp_start": 0.1,                  # ← starts late
            "tolerance_s": 0.002,                     # ← tight tolerance → violations
        },
        "observation.temperature": {
            "dtype": "float32", "shape": (1,), "names": ["temp"],
            "fps": 10, "tolerance_s": 0.05,           # ← loose tolerance
        },
        "observation.state": {
            "dtype": "float32", "shape": (7,),
            "names": [f"j{i}" for i in range(7)],
            "fps": 30, "tolerance_s": 0.001,           # ← tight, same FPS → OK
        },
    },
    root=root, use_videos=False,
)

# Record — IMU starts 0.1s late, causing violations on first 3 frames
for i in range(60):
    t = i / FPS
    for _ in range(33):
        ds.add_frame("observation.imu", np.random.randn(6).astype(np.float32))
    if i % 3 == 0:
        ds.add_frame("observation.temperature", np.array([25.0 + i * 0.1], dtype=np.float32))
    ds.add_frame("observation.state", np.sin(t + np.arange(7) * 0.5).astype(np.float32))
    ds.add_frame("task", f"step_{i}")
ds.save_episode()

print("\n=== Alignment Report (meta/alignment_check.jsonl) ===\n")
report_path = root / "meta" / "alignment_check.jsonl"
if report_path.exists():
    for line in report_path.read_text().strip().split("\n"):
        r = json.loads(line)
        r.pop("details", None)  # details are verbose
        print(json.dumps(r, indent=2))
