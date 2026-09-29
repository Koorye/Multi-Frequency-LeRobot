"""Audio demo: microphone stored as WAV + timestamp parquet, read back by frame."""

import shutil
import wave
from pathlib import Path

import numpy as np

from mf_lerobot import MultiFrequencyLeRobotDataset

FPS = 30
SAMPLE_RATE = 16000
H, W = 240, 320

# ── Helpers ──

def _camera_frame(t):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:H, 0:W] / np.array([H, W])[:, None, None]
    img[..., 0] = (80 + 40 * np.sin(t * 0.5) * (0.3 + 0.7 * xx)).astype(np.uint8)
    img[..., 1] = (60 + 30 * np.cos(t * 0.3) * (0.3 + 0.7 * yy)).astype(np.uint8)
    img[..., 2] = (100 + 50 * np.sin(t * 0.7) * (0.5 + 0.5 * xx)).astype(np.uint8)
    return img


def _state(t):
    return np.array([
        0.5 + 0.1 * np.sin(t * 0.8), 0.0 + 0.1 * np.cos(t * 0.6),
        0.3 + 0.05 * np.sin(t * 1.0),
        0.2 * np.sin(t * 1.2), 0.15 * np.cos(t * 0.9), 0.1 * np.sin(t * 1.5),
        0.5 + 0.5 * np.sin(t * 0.7),
    ], dtype=np.float32)


def _audio_chunk(t, n):
    """Simulated mic chunk: a sweeping tone + noise, float32 in [-1, 1]."""
    tt = t + np.arange(n) / SAMPLE_RATE
    tone = 0.3 * np.sin(2 * np.pi * (440 + 100 * np.sin(t * 0.5)) * tt)
    return (tone + 0.05 * np.random.randn(n)).astype(np.float32)


# ── Write ──

root = Path(__file__).parent.parent / "data" / "audio_demo"
if root.exists():
    shutil.rmtree(root)

samples_per_frame = SAMPLE_RATE // FPS  # 533 samples ≈ 33ms per master frame

ds = MultiFrequencyLeRobotDataset.create(
    repo_id="robot_audio_demo", fps=FPS,
    features={
        "observation.images.cam": {
            "dtype": "video", "shape": (H, W, 3), "names": ["h", "w", "c"], "fps": FPS,
        },
        "observation.state": {
            "dtype": "float32", "shape": (7,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
            "fps": FPS,
        },
        # audio: one add_frame per master frame; timestamps auto = counter × 1/fps
        "observation.audio.mic": {
            "dtype": "audio",
            "shape": (samples_per_frame,),
            "names": ["samples"],
            "sample_rate": SAMPLE_RATE,
            "fps": FPS,
        },
    },
    root=root,
)

FRAMES = 60  # 2 seconds
for ep, task_name in enumerate(["reach_target", "pour_liquid"]):
    for i in range(FRAMES):
        t = i / FPS
        ds.add_frame("observation.audio.mic", _audio_chunk(t, samples_per_frame))
        ds.add_frame("task", task_name)          # master clock
        ds.add_frame("observation.state", _state(t))
        ds.add_frame("observation.images.cam", _camera_frame(t))
    ds.save_episode()

# ── Inspect storage ──

print("\nOn disk:")
wav_path = root / "audios" / "chunk-000" / "observation.audio.mic" / "episode_000000.wav"
with wave.open(str(wav_path), "rb") as wf:
    print(f"  {wav_path.relative_to(root)}: {wf.getnchannels()}ch, "
          f"{wf.getframerate()}Hz, {wf.getnframes()} samples "
          f"({wf.getnframes() / wf.getframerate():.2f}s)")
ts_path = root / "data" / "chunk-000" / "episode_000000" / "observation.audio.mic.parquet"
print(f"  {ts_path.relative_to(root)}: timestamp parquet index")

# ── Read back ──

print("\nReading back:")
ds = MultiFrequencyLeRobotDataset(repo_id="robot_audio_demo", root=root)
item = ds[10]
print(f"  ds[10] (t={item['timestamp'].item():.3f}s):")
print(f"    observation.audio.mic          → {tuple(item['observation.audio.mic'].shape)} "
      f"float32 in [{item['observation.audio.mic'].min():.2f}, "
      f"{item['observation.audio.mic'].max():.2f}]")
print(f"    observation.state              → {tuple(item['observation.state'].shape)}")
print(f"    observation.images.cam         → {tuple(item['observation.images.cam'].shape)}")

# Window mode: sample-precise slice + concat across chunk boundaries
audio = ds._features["observation.audio.mic"]
t10 = 10 / FPS
w = audio.read(0, t10, window=(-0.033, 0.0))
w33 = audio.read(0, 1.23, window=(-0.33, 0.0))
print(f"\n  window mode (timestamps are an exact 1/{FPS}s grid here):")
print(f"    read(0, {t10:.4f}, window=(-0.033, 0)) → {tuple(w.shape)} "
      f"(期望 ~{int(0.033 * SAMPLE_RATE)} 采样 = 33ms, 跨块拼接)")
print(f"    read(0, 1.23,   window=(-0.33, 0))   → {tuple(w33.shape)} "
      f"(期望 ~{int(0.33 * SAMPLE_RATE)} 采样 = 330ms)")

# window_overrides at load time — same data, wider window, no rewrite
ds_wide = MultiFrequencyLeRobotDataset(
    repo_id="robot_audio_demo", root=root,
    window_overrides={"observation.audio.mic": (-0.066, 0.0)},
)
print(f"    window_overrides=(-0.066, 0) 后 ds[10] → "
      f"{tuple(ds_wide[10]['observation.audio.mic'].shape)}")

# Nearest-chunk semantics: every master frame must resolve to its own chunk
audio = ds._features["observation.audio.mic"]
ts = audio._load_timestamps(0)
frame_indices = ds.hf_dataset["frame_index"]
errors = [
    abs(int(np.argmin(np.abs(ts - float(fi) / FPS))) - fi)
    for fi in frame_indices
]
print(f"\n  nearest-chunk check over {len(frame_indices)} frames: "
      f"{'OK — chunk i returned for frame i' if not any(errors) else f'{sum(e != 0 for e in errors)} mismatches'}")
print(f"  episode duration: {len(ts) / FPS:.2f}s = "
      f"{ts[-1] + samples_per_frame / SAMPLE_RATE:.2f}s of audio")
