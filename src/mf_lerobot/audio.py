"""AudioFeature: one microphone — buffer chunks, write WAV, write timestamp parquet."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .utils import DEFAULT_AUDIO_PATH, DEFAULT_DATA_PATH


class AudioFeature:
    """One microphone: buffer audio chunks per episode, save WAV + timestamp parquet.

    Mirrors VideoFeature: each ``add`` records one chunk (its timestamp goes
    into the parquet index, one row per chunk); ``save`` concatenates the
    chunks into a single PCM16 WAV per episode.  ``read`` resolves the chunk
    nearest a query timestamp through the parquet index, then slices exactly
    that chunk out of the WAV.

    Samples are float32 in [-1, 1].  A chunk is ``(n_samples,)`` for mono or
    ``(n_samples, channels)`` for multichannel audio.
    """

    def __init__(self, key: str, spec: dict, root: Path, window_override=None):
        self.key = key
        self.spec = spec
        self.root = root
        self._window_override = window_override
        self._sample_rate = int(spec.get("sample_rate", 16000))
        self._ep_idx = 0
        self._chunk_count = 0
        self._chunks: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._ts_cache: dict[int, np.ndarray] = {}
        self._offsets_cache: dict[int, np.ndarray] = {}
        self._data_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # ── Add / Save / Read ──

    def add(self, samples: np.ndarray | torch.Tensor, timestamp: float) -> None:
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim > 2:
            raise ValueError(
                f"audio chunk for '{self.key}' must be 1-D mono or 2-D "
                f"(n_samples, channels), got shape {samples.shape}"
            )
        self._chunks.append(samples)
        self._timestamps.append(timestamp)
        self._chunk_count += 1

    def save(self) -> None:
        if not self._chunks:
            return
        chunks_size = self.spec.get("chunks_size", 1000)
        ep_chunk = self._ep_idx // chunks_size
        n_samples = np.array([len(c) for c in self._chunks], dtype=np.int64)
        ts_arr = np.array(self._timestamps, dtype=np.float64)

        # Timestamp parquet — one row per chunk, same file layout as video
        table = pa.table({
            "timestamp": pa.array(ts_arr, type=pa.float64()),
            "episode_index": pa.array(
                np.full(len(ts_arr), self._ep_idx, dtype=np.int64), type=pa.int64()
            ),
            "num_samples": pa.array(n_samples, type=pa.int64()),
        })
        fpath = self.root / DEFAULT_DATA_PATH.format(
            episode_chunk=ep_chunk, episode_index=self._ep_idx, feature_key=self.key,
        )
        fpath.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, fpath, compression="snappy")

        # WAV
        wav_path = self.root / DEFAULT_AUDIO_PATH.format(
            episode_chunk=ep_chunk, audio_key=self.key, episode_index=self._ep_idx,
        )
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_wav(wav_path)

    def _write_wav(self, fpath: Path) -> None:
        interleaved, channels = self._to_pcm16(np.concatenate(self._chunks, axis=0))
        with wave.open(str(fpath), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # PCM16
            wf.setframerate(self._sample_rate)
            wf.writeframes(interleaved.tobytes())

    def _to_pcm16(self, samples: np.ndarray) -> tuple[np.ndarray, int]:
        """float32 [-1, 1] → interleaved int16; returns (flat array, n_channels)."""
        channels = 1 if samples.ndim == 1 else samples.shape[1]
        np.clip(samples, -1.0, 1.0, out=samples)
        ints = (samples * 32767.0).astype("<i2")
        return ints.reshape(-1), channels

    def compute_stats(self, sample_count: int = 10) -> dict | None:
        """Amplitude stats in [-1, 1] over sampled chunks (per channel)."""
        n = self._chunk_count
        if n == 0:
            return None
        indices = list(range(0, n, max(1, n // sample_count)))[:sample_count]
        arrays = [self._chunks[i] for i in indices if len(self._chunks[i]) > 0]
        if not arrays:
            return None
        stacked = np.stack(
            [a if a.ndim == 2 else a[:, None] for a in arrays], axis=0
        )  # [S, N, C]
        axes = (0, 1)  # reduce over sampled chunks and samples — keep channel
        total = sum(len(c) for c in self._chunks)
        return {
            "min": stacked.min(axis=axes),
            "max": stacked.max(axis=axes),
            "mean": stacked.mean(axis=axes),
            "std": stacked.std(axis=axes),
            "count": np.array([total]),
        }

    def next_episode(self):
        self._ep_idx += 1
        self._chunk_count = 0
        self._chunks = []
        self._timestamps = []

    def read(self, ep_idx: int, timestamp: float, window=None) -> torch.Tensor:
        """Read audio around ``timestamp``.

        Two modes, aligned with the video path:

        - nearest (default): the chunk whose stored timestamp is nearest to
          ``timestamp``, returned whole — the atomic unit added via
          ``add_frame``, analogous to one video frame.
        - window ``(start, end)``: every sample whose time lies in
          ``(timestamp + start, timestamp + end]``, sliced with sample
          precision across chunk boundaries and concatenated.

        Timing contract: a chunk's first sample sits at its parquet label
        and samples advance at exactly ``1/sample_rate``; consecutive
        chunks are back-to-back.  Real dropouts (missing content) simply
        shorten the returned segment — content is never stretched.
        """
        window = window if window is not None else self._resolve_window()
        if isinstance(window, (list, tuple)) and len(window) == 2:
            return self._read_window(ep_idx, timestamp,
                                     float(window[0]), float(window[1]))
        ts = self._load_timestamps(ep_idx)
        if len(ts) == 0:
            return self._empty()
        idx = int(np.argmin(np.abs(ts - timestamp)))
        offsets = self._load_offsets(ep_idx)
        return self._read_pcm(ep_idx, int(offsets[idx]),
                              int(offsets[idx + 1] - offsets[idx]))

    def load(self, ep_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Whole episode as ``(per-sample timestamps, values)``.

        Mirrors ``ParquetFeature.load`` so generic consumers (web visualizer,
        alignment checker) can treat audio like any other signal: timestamps
        are per sample (chunk label + sample offset / sample_rate), values are
        the full-episode PCM shaped ``(N, channels)`` in [-1, 1].
        """
        if ep_idx not in self._data_cache:
            ts = self._load_timestamps(ep_idx)
            offsets = self._load_offsets(ep_idx)
            n = int(offsets[-1])
            if len(ts) == 0 or n == 0:
                shape = self.spec.get("shape", (0,))
                channels = shape[1] if len(shape) > 1 else 1
                self._data_cache[ep_idx] = (
                    np.array([], dtype=np.float64),
                    np.zeros((0, channels), dtype=np.float32),
                )
            else:
                vals = self._read_pcm(ep_idx, 0, n).numpy()
                if vals.ndim == 1:
                    vals = vals[:, None]
                ns = np.diff(offsets)
                pts = (np.repeat(ts, ns)
                       + (np.arange(n, dtype=np.float64)
                          - np.repeat(offsets[:-1], ns)) / self._sample_rate)
                self._data_cache[ep_idx] = (pts, vals)
        return self._data_cache[ep_idx]

    def _resolve_window(self):
        return self._window_override or self.spec.get("window")

    def _read_window(self, ep_idx: int, timestamp: float, w0: float, w1: float):
        ts = self._load_timestamps(ep_idx)
        offsets = self._load_offsets(ep_idx)
        if len(ts) == 0 or w1 <= w0:
            return self._empty()
        rate = self._sample_rate
        n = len(ts)
        chunk_ns = np.diff(offsets)          # samples per chunk
        lo, hi = timestamp + w0, timestamp + w1

        # chunks whose label-time span can intersect (lo, hi];
        # the per-chunk clamps below reject the rest (window past either end)
        i0 = int(np.clip(np.searchsorted(ts, lo, side="right") - 1, 0, n - 1))
        i1 = int(np.clip(np.searchsorted(ts, hi, side="left"), 0, n - 1))

        ranges: list[list[int]] = []
        for i in range(i0, i1 + 1):
            nc = int(chunk_ns[i])
            # sample j of chunk i has time ts[i] + j/rate; keep (lo, hi]
            j0 = 0 if lo <= ts[i] else int(np.floor((lo - ts[i]) * rate)) + 1
            j1 = (nc if hi >= ts[i] + (nc - 1) / rate
                  else int(np.floor((hi - ts[i]) * rate)) + 1)
            j0, j1 = max(0, j0), max(0, min(j1, nc))
            if j1 > j0:
                ranges.append([int(offsets[i]) + j0, int(offsets[i]) + j1])

        # merge sample-adjacent ranges → usually one contiguous seek+read
        merged: list[list[int]] = []
        for s, e in ranges:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        parts = [self._read_pcm(ep_idx, s, e - s) for s, e in merged]
        if not parts:
            return self._empty()
        if len(parts) == 1:
            return parts[0]
        return torch.from_numpy(np.concatenate(
            [p.numpy() for p in parts], axis=0))

    def _read_pcm(self, ep_idx: int, start: int, length: int) -> torch.Tensor:
        """Read ``length`` samples at ``start`` from the episode WAV."""
        if length <= 0:
            return self._empty()
        ep_chunk = ep_idx // self.spec.get("chunks_size", 1000)
        wav_path = self.root / DEFAULT_AUDIO_PATH.format(
            episode_chunk=ep_chunk, audio_key=self.key, episode_index=ep_idx,
        )
        with wave.open(str(wav_path), "rb") as wf:
            channels = wf.getnchannels()
            wf.setpos(start)
            raw = wf.readframes(length)
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels)
        return torch.from_numpy(samples)

    def _empty(self) -> torch.Tensor:
        shape = self.spec.get("shape", (0,))
        return torch.zeros(shape if len(shape) <= 1 else (0, shape[1]),
                           dtype=torch.float32)

    def _load_timestamps(self, ep_idx: int) -> np.ndarray:
        self._load_index(ep_idx)
        return self._ts_cache[ep_idx]

    def _load_offsets(self, ep_idx: int) -> np.ndarray:
        self._load_index(ep_idx)
        return self._offsets_cache[ep_idx]

    def _load_index(self, ep_idx: int) -> None:
        if ep_idx in self._ts_cache:
            return
        ep_chunk = ep_idx // self.spec.get("chunks_size", 1000)
        fpath = self.root / DEFAULT_DATA_PATH.format(
            episode_chunk=ep_chunk, episode_index=ep_idx, feature_key=self.key,
        )
        if fpath.exists():
            table = pq.read_table(fpath)
            if len(table):
                self._ts_cache[ep_idx] = table.column("timestamp").to_numpy()
                cum = np.concatenate([
                    np.array([0], dtype=np.int64),
                    np.cumsum(table.column("num_samples").to_numpy(), dtype=np.int64),
                ])
                self._offsets_cache[ep_idx] = cum
                return
        self._ts_cache[ep_idx] = np.array([], dtype=np.float64)
        self._offsets_cache[ep_idx] = np.array([0], dtype=np.int64)
