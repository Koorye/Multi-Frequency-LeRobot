"""Multi-frequency LeRobot dataset.

Extends LeRobotDataset with per-field add_frame and per-feature parquet storage.
Each non-video feature is backed by a ParquetFeature; cameras by VideoFeature.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .checker import DatasetChecker
from .parquet import ParquetFeature
from .video import VideoFeature
from .index import MasterIndex
from .metadata import MultiFrequencyDatasetMetadata
from .utils import DEFAULT_FEATURES

logger = logging.getLogger(__name__)


class MultiFrequencyLeRobotDataset(LeRobotDataset):
    """LeRobot dataset with per-field add_frame for multi-frequency sensors."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms=None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        window_overrides: dict[str, tuple | str | None] | None = None,
    ):
        self.root = Path(root) if root else (
            Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
        )
        self.root.mkdir(exist_ok=True, parents=True)
        self.repo_id = repo_id
        self.tolerance_s = tolerance_s
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0

        self.meta = MultiFrequencyDatasetMetadata(self.repo_id, self.root)
        self.index = MasterIndex(self)
        self.hf_dataset = self.index.load_hf_dataset()
        if episodes is not None:
            self.hf_dataset = self.hf_dataset.select(
                [i for i, ep in enumerate(self.hf_dataset["episode_index"])
                 if ep in episodes]
            )

        if delta_timestamps is not None:
            from lerobot.datasets.utils import check_delta_timestamps, get_delta_indices
            check_delta_timestamps(delta_timestamps, self.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.fps)
        else:
            self.delta_indices = None

        self._window_overrides = window_overrides or {}
        self._counters: dict[str, int] = {}

        self._features: dict[str, ParquetFeature | VideoFeature] = {}
        for key, ft in self.meta.features.items():
            if key in DEFAULT_FEATURES:
                continue
            if ft.get("dtype") in ("video", "image"):
                self._features[key] = VideoFeature(
                    key, ft, self.root,
                    backend=video_backend or "pyav",
                    tolerance_s=tolerance_s,
                )
            else:
                self._features[key] = ParquetFeature(
                    key, ft, self.root,
                    window_overrides.get(key) if window_overrides else None,
                )

    def __len__(self):
        return self.num_frames

    @property
    def num_frames(self) -> int:
        return len(self.hf_dataset) if self.hf_dataset is not None else 0

    @property
    def fps(self):
        return self.meta.fps

    # ── Factory ──

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: float,
        features: dict,
        robot_type: str | None = None,
        root: str | Path | None = None,
        use_videos: bool = True,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        window_overrides: dict[str, tuple | str | None] | None = None,
    ) -> "MultiFrequencyLeRobotDataset":
        root = Path(root) if root else (
            Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
        )

        meta = MultiFrequencyDatasetMetadata.create(
            repo_id=repo_id, fps=fps, features=features,
            robot_type=robot_type, root=root, use_videos=use_videos,
        )

        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = root
        obj.meta = meta
        obj.tolerance_s = 1e-4
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.batch_encoding_size = batch_encoding_size
        obj.episodes_since_last_encoding = 0
        obj.image_writer = None
        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        obj.index = MasterIndex(obj)
        obj._window_overrides = window_overrides or {}
        obj._counters = {}

        obj._features = {}
        for key, ft in meta.features.items():
            if key in DEFAULT_FEATURES:
                continue
            if ft.get("dtype") in ("video", "image"):
                obj._features[key] = VideoFeature(
                    key, ft, obj.root,
                    backend=video_backend or "pyav",
                    tolerance_s=1e-4,
                )
            else:
                obj._features[key] = ParquetFeature(
                    key, ft, obj.root,
                    window_overrides.get(key) if window_overrides else None,
                )

        obj.hf_dataset = obj.create_hf_dataset()
        return obj

    # ── Index delegation ──

    def get_episodes_file_paths(self) -> list[str]:
        return self.index.file_paths()

    @property
    def hf_features(self):
        return self.index.hf_features()

    def load_hf_dataset(self):
        return self.index.load_hf_dataset()

    # ── Core API ──

    def add_frame(
        self,
        key: str,
        value: np.ndarray | torch.Tensor | str,
        timestamp: float | None = None,
    ) -> None:
        if key == "task":
            if timestamp is None:
                timestamp = len(self.index.records) / self.fps
            self.index.record_frame(timestamp, str(value))
            return

        f = self._features.get(key)
        if f is None:
            raise ValueError(f"Unknown feature: '{key}'")

        if timestamp is None:
            timestamp = self._auto_timestamp(key)

        f.add(value, timestamp)

    def _auto_timestamp(self, key: str) -> float:
        ft = self.meta.features.get(key, {})
        fps = self.meta.get_feature_fps(key) or ft.get("fps") or self.fps
        start = ft.get("timestamp_start", 0.0)
        if key not in self._counters:
            self._counters[key] = 0
        ts = start + self._counters[key] * (1.0 / fps)
        self._counters[key] += 1
        return ts

    # ── Save ──

    def save_episode(self) -> None:
        ep_idx = self.meta.total_episodes
        episode_length = len(self.index.records)
        if episode_length == 0:
            logger.warning("No frames recorded; skipping save_episode")
            return

        # Task registration
        tasks = [r["task"] for r in self.index.records]
        episode_tasks = list(dict.fromkeys(tasks))
        for task in episode_tasks:
            if self.meta.get_task_index(task) is None:
                self.meta.add_task(task)

        # Write index
        index_table = self.index.write(ep_idx)

        # Save all features
        chunks_size = self.meta.info.get("chunks_size", 1000)
        has_video = False
        for f in self._features.values():
            if isinstance(f, ParquetFeature):
                f.save(chunks_size)
            elif isinstance(f, VideoFeature):
                f.save()
                has_video = True

        if has_video and ep_idx == 0:
            self.meta.update_video_info()

        # Stats + metadata
        ep_stats = {}
        for f in self._features.values():
            s = f.compute_stats()
            if s:
                ep_stats[f.key] = s
        self.meta.save_episode(ep_idx, episode_length, episode_tasks, ep_stats)
        self.index.append_to_memory(index_table)

        # Checks
        checker = DatasetChecker(self)
        if has_video:
            checker.check_video_frames(ep_idx, episode_length, self.index.records)
        checker.check_episode_alignment(ep_idx, episode_length, self.index.records)

        # Batch encoding — native lerobot encodes every N episodes
        if has_video and self.batch_encoding_size > 1:
            self.episodes_since_last_encoding += 1
            if self.episodes_since_last_encoding >= self.batch_encoding_size:
                start_ep = ep_idx - self.episodes_since_last_encoding + 1
                end_ep = ep_idx + 1
                logger.info(f"Batch encoding videos for episodes {start_ep}-{end_ep - 1}")
                self.episodes_since_last_encoding = 0

        # Reset
        self.index.reset()
        self._counters.clear()
        for f in self._features.values():
            f.next_episode()
        logger.info(f"Episode {ep_idx}: {episode_length} frames")

    # ── Read ──

    def __getitem__(self, idx: int) -> dict:
        row = self.hf_dataset[idx]
        ep = row["episode_index"].item()
        ts = row["timestamp"].item()
        task_idx = row["task_index"].item()

        item = {
            "timestamp": row["timestamp"],
            "frame_index": row["frame_index"],
            "episode_index": row["episode_index"],
            "index": row["index"],
            "task_index": row["task_index"],
            "task": self.meta.tasks.get(task_idx, ""),
        }

        # Compute query timestamps from delta_indices
        query_ts_list = [ts]
        if self.delta_indices is not None:
            for key, offsets in self.delta_indices.items():
                query_ts_list = [ts + o / self.fps for o in offsets]

        for f in self._features.values():
            if isinstance(f, VideoFeature):
                frame = f.read(ep, ts)
                if self.image_transforms is not None:
                    frame = self.image_transforms(frame)
                item[f.key] = frame
            else:
                if self.delta_indices is not None:
                    results = [f.query(ep, t) for t in query_ts_list]
                    for k in results[0]:
                        item[k] = torch.stack([r[k] for r in results])
                else:
                    item.update(f.query(ep, ts))
        return item
