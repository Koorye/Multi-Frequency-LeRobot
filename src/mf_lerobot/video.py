"""VideoFeature: one camera — save images, encode MP4, write timestamp parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from .utils import DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH

DEFAULT_IMAGE_PATH = "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png"


class VideoFeature:
    """One camera: save temp images, encode MP4, write timestamp parquet."""

    def __init__(
        self, key: str, spec: dict, root: Path,
        backend: str = "pyav", tolerance_s: float = 0.1,
    ):
        self.key = key
        self.spec = spec
        self.root = root
        self._backend = backend
        self._tolerance_s = tolerance_s
        self._ep_idx = 0
        self._frame_count = 0
        self._timestamps: list[float] = []
        self._ts_cache: dict[int, np.ndarray] = {}

    # ── Image path ──

    def _image_path(self, frame_index: int) -> Path:
        return self.root / DEFAULT_IMAGE_PATH.format(
            image_key=self.key, episode_index=self._ep_idx, frame_index=frame_index,
        )

    def _save_image(self, image: np.ndarray, fpath: Path) -> None:
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32 and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            img = Image.fromarray(image)
        else:
            img = image
        img.save(fpath)

    # ── Add / Save / Read ──

    def add(self, image: np.ndarray, timestamp: float) -> None:
        fi = self._frame_count
        path = self._image_path(fi)
        if fi == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._save_image(image, path)
        self._timestamps.append(timestamp)
        self._frame_count += 1

    def save(self) -> None:
        from lerobot.datasets.video_utils import encode_video_frames

        chunks_size = self.spec.get("chunks_size", 1000)
        ep_chunk = self._ep_idx // chunks_size
        ts_arr = np.array(self._timestamps, dtype=np.float64)

        # Timestamp parquet
        table = pa.table({
            "timestamp": pa.array(ts_arr, type=pa.float64()),
            "episode_index": pa.array(
                np.full(len(ts_arr), self._ep_idx, dtype=np.int64), type=pa.int64()
            ),
        })
        fpath = self.root / DEFAULT_DATA_PATH.format(
            episode_chunk=ep_chunk, episode_index=self._ep_idx, feature_key=self.key,
        )
        fpath.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, fpath, compression="snappy")

        # MP4
        video_path = self.root / DEFAULT_VIDEO_PATH.format(
            episode_chunk=ep_chunk, video_key=self.key, episode_index=self._ep_idx,
        )
        if not video_path.is_file():
            encode_video_frames(
                self._image_path(0).parent, video_path,
                self.spec.get("fps", 30), overwrite=True,
            )

    def compute_stats(self, sample_count: int = 10) -> dict | None:
        """Compute per-channel pixel stats from sampled frame images,
        matching lerobot's native video stats format (values in [0, 1])."""
        n = self._frame_count
        if n == 0:
            return None
        indices = list(range(0, n, max(1, n // sample_count)))[:sample_count]
        arrays = []
        for fi in indices:
            path = self._image_path(fi)
            if path.exists():
                img = np.array(Image.open(path))
                if img.ndim == 2:
                    img = img[..., None]
                arrays.append(img.astype(np.float32) / 255.0)
        if not arrays:
            return None
        stacked = np.stack(arrays, axis=0)  # [S, H, W, C]
        stacked = np.transpose(stacked, (0, 3, 1, 2))  # [S, C, H, W]
        axes = (0, 2, 3)  # reduce over samples, height, width — keep channel
        # Lerobot format: (C, 1, 1) — squeeze the batch dim
        return {
            "min": stacked.min(axis=axes, keepdims=True).squeeze(0),
            "max": stacked.max(axis=axes, keepdims=True).squeeze(0),
            "mean": stacked.mean(axis=axes, keepdims=True).squeeze(0),
            "std": stacked.std(axis=axes, keepdims=True).squeeze(0),
            "count": np.array([n]),
        }

    def next_episode(self):
        self._ep_idx += 1
        self._frame_count = 0
        self._timestamps = []

    def read(self, ep_idx: int, timestamp: float) -> torch.Tensor:
        """Frame whose stored timestamp is nearest to ``timestamp``.

        The video itself is constant-fps, so its pts sit on a k/fps grid while
        real sensor timestamps jitter around it — decoding by timestamp alone
        fails tiny tolerances.  Resolve the frame through the timestamps
        parquet first (same semantics as lerobot's native video decoder),
        then decode that exact grid position.
        """
        ep_chunk = ep_idx // self.spec.get("chunks_size", 1000)
        video_path = self.root / DEFAULT_VIDEO_PATH.format(
            episode_chunk=ep_chunk, video_key=self.key, episode_index=ep_idx,
        )
        ts = self._load_timestamps(ep_idx)
        if len(ts) == 0:
            shape = self.spec.get("shape", (0, 0, 3))
            return torch.zeros(3, shape[0], shape[1], dtype=torch.uint8)
        fps = float(self.spec.get("fps", 30) or 30)
        idx = int(np.argmin(np.abs(ts - timestamp)))
        from lerobot.datasets.video_utils import decode_video_frames
        frames = decode_video_frames(
            video_path, [idx / fps],
            tolerance_s=max(self._tolerance_s, 0.5 / fps), backend=self._backend,
        )
        return frames.squeeze(0) if isinstance(frames, torch.Tensor) else torch.from_numpy(frames[0])

    def _load_timestamps(self, ep_idx: int) -> np.ndarray:
        if ep_idx not in self._ts_cache:
            ep_chunk = ep_idx // self.spec.get("chunks_size", 1000)
            fpath = self.root / DEFAULT_DATA_PATH.format(
                episode_chunk=ep_chunk, episode_index=ep_idx, feature_key=self.key,
            )
            if fpath.exists():
                table = pq.read_table(fpath)
                self._ts_cache[ep_idx] = (
                    table.column("timestamp").to_numpy()
                    if len(table) else np.array([], dtype=np.float64)
                )
            else:
                self._ts_cache[ep_idx] = np.array([], dtype=np.float64)
        return self._ts_cache[ep_idx]
