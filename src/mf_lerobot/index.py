"""Master frame index — bridges per-feature storage with LeRobotDataset.

A lightweight parquet per episode with columns:
    timestamp | frame_index | episode_index | index | task_index
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .utils import DEFAULT_FEATURES, DEFAULT_INDEX_PATH


class MasterIndex:
    """Manages the frame index that LeRobotDataset reads from."""

    def __init__(self, dataset):
        self.ds = dataset
        self.records: list[dict[str, Any]] = []  # {timestamp, task, frame_index}

    # ── Record frames ──

    def record_frame(self, timestamp: float, task: str) -> int:
        """Record a master frame. Returns the frame index within the episode."""
        frame_index = len(self.records)
        self.records.append({
            "timestamp": timestamp,
            "task": task or "default_task",
            "frame_index": frame_index,
        })
        return frame_index

    def reset(self):
        self.records = []

    # ── Write ──

    def write(
        self,
        ep_idx: int,
    ) -> pa.Table:
        """Build and write master_index.parquet. Returns the table for HF append."""
        episode_length = len(self.records)
        chunks_size = self.ds.meta.info.get("chunks_size", 1000)
        ep_chunk = ep_idx // chunks_size

        frame_ts = np.array(
            [r["timestamp"] for r in self.records], dtype=np.float32
        )
        frame_indices_arr = np.arange(episode_length, dtype=np.int64)
        ep_arr = np.full(episode_length, ep_idx, dtype=np.int64)
        global_start = self.ds.meta.total_frames
        global_indices = np.arange(
            global_start, global_start + episode_length, dtype=np.int64
        )

        # task indices must be set by caller after task registration
        tasks = [r["task"] for r in self.records]
        task_indices = np.array(
            [self.ds.meta.get_task_index(t) for t in tasks], dtype=np.int64
        )

        table = pa.table({
            "timestamp": pa.array(frame_ts, type=pa.float32()),
            "frame_index": pa.array(frame_indices_arr, type=pa.int64()),
            "episode_index": pa.array(ep_arr, type=pa.int64()),
            "index": pa.array(global_indices, type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64()),
        })

        path = self.ds.root / DEFAULT_INDEX_PATH.format(
            episode_chunk=ep_chunk, episode_index=ep_idx
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="snappy")
        return table

    # ── Read ──

    def load_hf_dataset(self):
        """Load all master_index parquets into an HF Dataset."""
        from datasets import load_dataset as hf_load_dataset
        from lerobot.datasets.utils import hf_transform_to_torch
        import glob

        pattern = str(self.ds.root / "data/chunk-*/episode_*/master_index.parquet")
        files = sorted(glob.glob(pattern))
        if files:
            hf_dataset = hf_load_dataset("parquet", data_files=files, split="train")
        else:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            hf_dataset = LeRobotDataset.create_hf_dataset(self.ds)
            hf_dataset.set_transform(hf_transform_to_torch)
            return hf_dataset
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def file_paths(self) -> list[str]:
        import glob
        pattern = str(self.ds.root / "data/chunk-*/episode_*/master_index.parquet")
        return sorted(glob.glob(pattern))

    def hf_features(self):
        from lerobot.datasets.utils import get_hf_features_from_features
        index_features = {
            k: v for k, v in self.ds.meta.features.items() if k in DEFAULT_FEATURES
        }
        return get_hf_features_from_features(index_features)

    # ── In-memory append ──

    def append_to_memory(self, index_table: pa.Table) -> None:
        from datasets import Dataset, concatenate_datasets
        from lerobot.datasets.utils import hf_transform_to_torch

        ep_dict = {
            col: index_table.column(col).to_pylist()
            for col in index_table.column_names
        }
        ep_dataset = Dataset.from_dict(
            ep_dict, features=self.hf_features(), split="train"
        )
        ep_dataset.set_transform(hf_transform_to_torch)
        self.ds.hf_dataset = concatenate_datasets([self.ds.hf_dataset, ep_dataset])
        self.ds.hf_dataset.set_transform(hf_transform_to_torch)
