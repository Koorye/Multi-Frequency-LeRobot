"""Realistic demo: 3 cameras + IMU + EEG + EMG + temperature, 14-DoF end-effector."""

import shutil
from pathlib import Path
import numpy as np
from mf_lerobot import MultiFrequencyLeRobotDataset

FPS = 30
H, W = 480, 640
EEG_CH = 8
EMG_CH = 4
EE_DOF = 7          # x, y, z, roll, pitch, yaw, gripper

# ── Helpers ──

def _camera_frame(t, cam_id):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:H, 0:W] / np.array([H, W])[:, None, None]

    bg_r = int(80 + 40 * np.sin(t * 0.5))
    bg_g = int(60 + 30 * np.cos(t * 0.3))
    bg_b = int(100 + 50 * np.sin(t * 0.7 + 1))
    img[..., 0] = (bg_r * (0.3 + 0.7 * xx)).astype(np.uint8)
    img[..., 1] = (bg_g * (0.3 + 0.7 * yy)).astype(np.uint8)
    img[..., 2] = (bg_b * (0.5 + 0.5 * (xx + yy) / 2)).astype(np.uint8)

    # Moving robot arm ellipse
    cx = W // 2 + int(80 * np.sin(t * 1.2 + cam_id * 2.1))
    cy = H // 2 + int(60 * np.cos(t * 0.8 + cam_id * 1.7))
    rx, ry = 120, 40
    angle = t * 0.6 + cam_id
    ca, sa = np.cos(angle), np.sin(angle)
    for dy in range(-ry, ry + 1, 2):
        for dx in range(-rx, rx + 1, 2):
            xr = int(cx + dx * ca - dy * sa); yr = int(cy + dx * sa + dy * ca)
            if 0 <= xr < W and 0 <= yr < H:
                img[yr, xr] = np.clip(img[yr, xr].astype(int) + [60, 50, 40], 0, 255).astype(np.uint8)

    # Target circle
    tx = W // 3 + int(40 * np.cos(t * 0.4 + cam_id * 1.3))
    ty = H // 3 + int(30 * np.sin(t * 0.6 + cam_id * 0.9))
    rr = 25 + int(10 * np.sin(t * 2.0))
    mask = ((np.arange(W)[None, :] - tx) ** 2 + (np.arange(H)[:, None] - ty) ** 2) <= rr ** 2
    img[mask] = np.clip(img[mask].astype(int) + [0, 80, 100], 0, 255).astype(np.uint8)
    return img


def _imu(t):
    return np.array([
        0.3 * np.sin(t * 3.0) + np.random.randn() * 0.02,
        0.2 * np.cos(t * 2.5 + 0.5) + np.random.randn() * 0.02,
        -9.81 + 0.5 * np.sin(t * 1.5) + np.random.randn() * 0.03,
        0.1 * np.sin(t * 4.0 + 1.0) + np.random.randn() * 0.005,
        0.08 * np.cos(t * 3.5) + np.random.randn() * 0.005,
        0.05 * np.sin(t * 2.0) + np.random.randn() * 0.003,
    ], dtype=np.float32)


def _eeg(t, ch=EEG_CH):
    """8ch EEG with alpha (10Hz), beta (20Hz), gamma (40Hz) bands + noise."""
    raw = np.zeros(ch, dtype=np.float32)
    for c in range(ch):
        raw[c] = (
            5.0 * np.sin(2 * np.pi * 10 * t + c * 0.8) * (1 + 0.3 * np.sin(t * 0.5)) +   # alpha
            2.0 * np.sin(2 * np.pi * 20 * t + c * 1.2) * (1 + 0.2 * np.cos(t * 0.3)) +   # beta
            1.0 * np.sin(2 * np.pi * 40 * t + c * 0.5) * (0.5 + 0.5 * np.sin(t * 1.0)) + # gamma
            np.random.randn() * 1.5
        )
    return raw


def _emg(t, task_phase, ch=EMG_CH):
    """4ch EMG: burst activity during movement phases, quiet otherwise."""
    activity = 0.5 + 0.5 * np.sin(task_phase * np.pi)  # 0→1→0 over phase
    raw = np.zeros(ch, dtype=np.float32)
    for c in range(ch):
        burst = activity * (0.3 + 0.2 * c) * np.abs(np.sin(t * 15 + c * 2.0))
        raw[c] = burst * np.exp(-0.5 * ((t * 20) % 2)) + np.random.randn() * 0.02
    return raw


def _ee_pose(t, offsets):
    """14-DoF end-effector: [x, y, z, roll, pitch, yaw, gripper]."""
    x = 0.5 + 0.15 * np.sin(t * 0.8 + offsets[0])
    y = 0.0 + 0.10 * np.cos(t * 0.6 + offsets[1])
    z = 0.3 + 0.08 * np.sin(t * 1.0 + offsets[2])
    roll  = 0.2 * np.sin(t * 1.2 + offsets[3])
    pitch = 0.15 * np.cos(t * 0.9 + offsets[4])
    yaw   = 0.1 * np.sin(t * 1.5 + offsets[5])
    grip  = 0.5 + 0.5 * np.sin(t * 0.7 + offsets[6])  # 0=open, 1=close
    return np.array([x, y, z, roll, pitch, yaw, grip], dtype=np.float32)


# ── Dataset ──

root = Path(__file__).parent.parent / "data" / "demo"
if root.exists():
    shutil.rmtree(root)

ds = MultiFrequencyLeRobotDataset.create(
    repo_id="robot_demo", fps=FPS,
    features={
        "observation.images.head_rgb": {
            "dtype": "video", "shape": (H, W, 3), "names": ["h", "w", "c"],
            "fps": 30, "tolerance_s": 0.001,
        },
        "observation.images.left_wrist_rgb": {
            "dtype": "video", "shape": (H, W, 3), "names": ["h", "w", "c"],
            "fps": 30, "tolerance_s": 0.001,
        },
        "observation.images.right_wrist_rgb": {
            "dtype": "video", "shape": (H, W, 3), "names": ["h", "w", "c"],
            "fps": 30, "tolerance_s": 0.001,
        },
        "observation.imu": {
            "dtype": "float32", "shape": (6,),
            "names": ["ax", "ay", "az", "gx", "gy", "gz"],
            "fps": 1000, "window": (-0.033, 0.0),
            "timestamp_start": 0.1, "tolerance_s": 0.002,
        },
        "observation.eeg": {
            "dtype": "float32", "shape": (EEG_CH,),
            "names": ["Fz", "Cz", "Pz", "Oz", "F3", "F4", "C3", "C4"],
            "fps": 256, "window": (-0.033, 0.0),
            "tolerance_s": 0.004,
        },
        "observation.emg_left": {
            "dtype": "float32", "shape": (EMG_CH,),
            "names": ["flex_carpi", "ext_carpi", "biceps", "triceps"],
            "fps": 100, "window": (-0.033, 0.0),
            "tolerance_s": 0.01,
        },
        "observation.emg_right": {
            "dtype": "float32", "shape": (EMG_CH,),
            "names": ["flex_carpi", "ext_carpi", "biceps", "triceps"],
            "fps": 100, "window": (-0.033, 0.0),
            "tolerance_s": 0.01,
        },
        "observation.temperature": {
            "dtype": "float32", "shape": (1,), "names": ["temp"],
            "fps": 10, "tolerance_s": 0.05,
        },
        "observation.state": {
            "dtype": "float32", "shape": (EE_DOF,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
            "fps": 30, "tolerance_s": 0.001,
        },
        "action": {
            "dtype": "float32", "shape": (EE_DOF,),
            "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "dgrip"],
            "fps": 30, "tolerance_s": 0.001,
        },
    },
    root=root, use_videos=True,
)

# ── Record ──

TASKS = ["reach_target", "pour_liquid", "stack_blocks"]
DURATION = 2.0
FRAMES = int(FPS * DURATION)
CAMERAS = [
    "observation.images.head_rgb",
    "observation.images.left_wrist_rgb",
    "observation.images.right_wrist_rgb",
]

for ep, task_name in enumerate(TASKS):
    ep_offset = ep * 10.0
    base_temp = 25.0 + ep * 3.0
    ee_offsets = np.linspace(ep_offset, ep_offset + np.pi, EE_DOF)

    for i in range(FRAMES):
        t = i / FPS
        abs_t = t + ep * DURATION
        phase = i / FRAMES  # 0→1 within episode

        # High-frequency sensors
        for _ in range(1000 // FPS):
            ds.add_frame("observation.imu", _imu(abs_t))
        for _ in range(256 // FPS):
            ds.add_frame("observation.eeg", _eeg(abs_t))
        for _ in range(100 // FPS):
            ds.add_frame("observation.emg_left", _emg(abs_t, phase))
            ds.add_frame("observation.emg_right", _emg(abs_t + 0.1, phase))

        # Temperature
        if i % 3 == 0:
            temp = base_temp + 2.0 * np.sin(t * 0.3) + np.random.randn() * 0.05
            ds.add_frame("observation.temperature", np.array([temp], dtype=np.float32))

        # Master clock
        ds.add_frame("task", task_name)

        # EE state + action
        ds.add_frame("observation.state", _ee_pose(t, ee_offsets))
        ds.add_frame("action", _ee_pose(t + 0.05, ee_offsets + 0.1))

        # Cameras
        for cid, cam in enumerate(CAMERAS):
            ds.add_frame(cam, _camera_frame(t, cid))

    ds.save_episode()
    print(f"Episode {ep} '{task_name}': {ds.num_frames} total frames")
