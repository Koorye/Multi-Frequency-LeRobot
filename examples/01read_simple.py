"""Read demo: show alignment strategies + cameras."""
from pathlib import Path
from mf_lerobot import MultiFrequencyLeRobotDataset

root = Path(__file__).parent.parent / "data" / "demo"
ds = MultiFrequencyLeRobotDataset(repo_id="minimal_demo", root=root, video_backend="pyav")

print(f"Frames: {ds.num_frames}\n")

item = ds[10]  # pick a middle frame
ts = item["timestamp"].item()

# ── window=(-0.033, 0.0) ──  all readings in (t-33ms, t]
imu = item["observation.imu"]
imu_ts = item["observation.imu_timestamps"]
print(f"[window=(-0.033,0)]  imu: {imu.shape} readings, "
      f"timestamps {imu_ts.shape}, "
      f"range [{imu_ts[0].item():.4f}, {imu_ts[-1].item():.4f}]")

# ── default (nearest) ──
temp = item["observation.temperature"]
print(f"[nearest]            temperature: {temp.item():.1f}°C")

# ── Override window at read time ──
ds2 = MultiFrequencyLeRobotDataset(
    repo_id="minimal_demo", root=root, video_backend="pyav",
    window_overrides={"observation.imu": (-0.066, 0.0)},
)
item2 = ds2[10]
imu2 = item2["observation.imu"]
imu2_ts = item2["observation.imu_timestamps"]
print(f"\n[override window=(-0.066,0)] imu: {imu2.shape} readings, "
      f"range [{imu2_ts[0].item():.4f}, {imu2_ts[-1].item():.4f}]")

# ── cameras ──
print(f"\n[camera]        head_rgb/left_wrist_rgb/right_wrist_rgb: "
      f"{item['observation.images.head_rgb'].shape} each")

# ── master-clock ──
print(f"[master]        state: {item['observation.state'].shape}, "
      f"action: {item['action'].shape}")

# IMU segment sizes across frames
print(f"\nIMU segment sizes (frames 0-9): "
      f"{[ds[i]['observation.imu'].shape[0] for i in range(10)]}")

# Temperature repeat: same value across frames
print(f"Temperature across frames 0-30 (every 3): "
      f"{[ds[i]['observation.temperature'].item() for i in range(0, 31, 3)]}")
