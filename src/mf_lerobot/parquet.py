"""ParquetFeature: one non-video feature — buffer, write parquet, load, query."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .utils import DEFAULT_DATA_PATH


class ParquetFeature:
    """One parquet-stored feature: write + read + window-based query."""

    def __init__(
        self,
        key: str,
        spec: dict,
        root: Path,
        window_override=None,
    ):
        self.key = key
        self.spec = spec
        self.root = root
        self._window_override = window_override
        self._ep_idx = 0
        self.buffer: list[tuple[float, np.ndarray]] = []
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # ── Write ──

    def add(self, value: np.ndarray | torch.Tensor, timestamp: float) -> float:
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        self.buffer.append((timestamp, value))
        return timestamp

    def save(self, chunks_size: int = 1000) -> dict[str, int | float]:
        if not self.buffer:
            return {}
        timestamps = np.array([s[0] for s in self.buffer], dtype=np.float64)
        values = np.stack([s[1] for s in self.buffer], axis=0)

        ep_chunk = self._ep_idx // chunks_size
        fpath = self.root / DEFAULT_DATA_PATH.format(
            episode_chunk=ep_chunk, episode_index=self._ep_idx, feature_key=self.key,
        )
        fpath.parent.mkdir(parents=True, exist_ok=True)

        table = _build_table(timestamps, self._ep_idx, values, _column_names(self.spec))
        pq.write_table(table, fpath, compression="snappy")

        from .utils import (
            feature_chunk_key, feature_file_key,
            feature_from_ts_key, feature_to_ts_key, feature_length_key,
        )
        return {
            feature_chunk_key(self.key): ep_chunk,
            feature_file_key(self.key): 0,
            feature_from_ts_key(self.key): float(timestamps[0]),
            feature_to_ts_key(self.key): float(timestamps[-1]),
            feature_length_key(self.key): len(timestamps),
        }

    def compute_stats(self) -> dict | None:
        if not self.buffer:
            return None
        vals = np.stack([s[1] for s in self.buffer], axis=0)
        return {
            "min": vals.min(axis=0), "max": vals.max(axis=0),
            "mean": vals.mean(axis=0), "std": vals.std(axis=0),
            "count": np.array([len(vals)]),
        }

    def next_episode(self):
        self._ep_idx += 1
        self.buffer = []

    # ── Read ──

    def load(self, ep_idx: int, chunks_size: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        if ep_idx not in self._cache:
            ep_chunk = ep_idx // chunks_size
            fpath = self.root / DEFAULT_DATA_PATH.format(
                episode_chunk=ep_chunk, episode_index=ep_idx, feature_key=self.key,
            )
            if fpath.exists():
                table = pq.read_table(fpath)
                if len(table) > 0:
                    ts = table.column("timestamp").to_numpy()
                    val_cols = [c for c in table.column_names
                                if c not in ("timestamp", "episode_index")]
                    vals = np.column_stack(
                        [table.column(c).to_numpy() for c in val_cols]
                    ).astype(np.float32)
                    self._cache[ep_idx] = (ts, vals)
                    return ts, vals
            D = self.spec.get("shape", (1,))[0]
            self._cache[ep_idx] = (
                np.array([], dtype=np.float64),
                np.full((0, D), np.nan, dtype=np.float32),
            )
        return self._cache[ep_idx]

    def query(self, ep_idx: int, t_current: float) -> dict:
        ts, vals = self.load(ep_idx)
        window = self._window_override or self.spec.get("window")
        result = {}

        if isinstance(window, (list, tuple)) and len(window) == 2:
            seg_ts, seg_vals = _query_range(ts, vals, t_current + window[0],
                                            t_current + window[1])
            result[self.key] = torch.from_numpy(seg_vals).float()
            result[f"{self.key}_timestamps"] = torch.from_numpy(seg_ts).float()
        elif window == "interpolate":
            result[self.key] = torch.from_numpy(_query_interpolate(ts, vals, t_current)).float()
        else:
            result[self.key] = torch.from_numpy(_query_nearest(ts, vals, t_current)).float()
        return result


# ── Internal ──

def _build_table(ts, ep_idx, vals, cols) -> pa.Table:
    N, D = vals.shape
    columns = {
        "timestamp": pa.array(ts, type=pa.float64()),
        "episode_index": pa.array(np.full(N, ep_idx, dtype=np.int64), type=pa.int64()),
    }
    for j, name in enumerate(cols):
        columns[name] = pa.array(vals[:, j], type=pa.float32())
    return pa.table(columns)


def _column_names(spec: dict) -> list[str]:
    names = spec.get("names")
    return list(names) if names else [f"dim_{i}" for i in range(spec.get("shape", (1,))[0])]


def _query_range(ts, vals, t_start, t_end):
    if len(ts) == 0:
        return np.array([], dtype=ts.dtype), np.zeros((0, vals.shape[1]), dtype=vals.dtype)
    mask = (ts > t_start) & (ts <= t_end)
    return ts[mask], vals[mask]


def _query_nearest(ts, vals, t_query):
    if len(ts) == 0:
        return np.full(vals.shape[1], np.nan, dtype=vals.dtype)
    return vals[np.argmin(np.abs(ts - t_query))]


def _query_interpolate(ts, vals, t_query):
    if len(ts) == 0:
        return np.full(vals.shape[1], np.nan, dtype=vals.dtype)
    if len(ts) == 1:
        return vals[0]
    idx = np.searchsorted(ts, t_query)
    if idx == 0:
        return vals[0]
    if idx >= len(ts):
        return vals[-1]
    t_a, t_b = ts[idx - 1], ts[idx]
    v_a, v_b = vals[idx - 1], vals[idx]
    if t_b == t_a:
        return v_a
    return v_a + (t_query - t_a) / (t_b - t_a) * (v_b - v_a)
