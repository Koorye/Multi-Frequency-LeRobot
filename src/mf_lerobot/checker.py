"""Post-save validation: video frame count and timestamp alignment."""

from __future__ import annotations

from typing import Any

import numpy as np

from lerobot.datasets.utils import append_jsonlines

from .utils import DEFAULT_AUDIO_PATH, DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH


class DatasetChecker:
    """Validates saved episodes: video integrity and timestamp alignment."""

    def __init__(self, dataset):
        self.ds = dataset

    def check_video_frames(
        self, ep_idx: int, episode_length: int, frame_records: list[dict[str, Any]]
    ) -> None:
        chunks_size = self.ds.meta.info.get("chunks_size", 1000)
        ep_chunk = ep_idx // chunks_size
        for cam_key in self.ds.meta.camera_keys:
            video_path = self.ds.root / DEFAULT_VIDEO_PATH.format(
                episode_chunk=ep_chunk, video_key=cam_key, episode_index=ep_idx
            )
            if not video_path.exists():
                print(f"[VIDEO] episode {ep_idx}, '{cam_key}': MISSING")
                continue
            try:
                import av
                container = av.open(str(video_path))
                video_stream = next(s for s in container.streams if s.type == "video")
                nb_frames = video_stream.frames or 0
                fps_stream = float(video_stream.average_rate or 0)
                time_base = float(video_stream.time_base)
                duration_s = float(video_stream.duration or 0) * time_base
                container.close()
                if nb_frames != episode_length:
                    print(f"[VIDEO] episode {ep_idx}, '{cam_key}': "
                          f"frame count {nb_frames} != {episode_length}")
                else:
                    print(f"[VIDEO] episode {ep_idx}, '{cam_key}': "
                          f"OK ({nb_frames}f, {fps_stream:.1f}fps, "
                          f"duration {duration_s:.2f}s)")
            except ImportError:
                pass
            except Exception as e:
                print(f"[VIDEO] episode {ep_idx}, '{cam_key}': error — {e}")

    def check_audio_frames(
        self, ep_idx: int, episode_length: int, frame_records: list[dict[str, Any]]
    ) -> None:
        """WAV integrity: chunk count vs index rows, sample count vs summed index."""
        import wave

        chunks_size = self.ds.meta.info.get("chunks_size", 1000)
        ep_chunk = ep_idx // chunks_size
        for audio_key in self._audio_keys():
            wav_path = self.ds.root / DEFAULT_AUDIO_PATH.format(
                episode_chunk=ep_chunk, audio_key=audio_key, episode_index=ep_idx
            )
            if not wav_path.exists():
                print(f"[AUDIO] episode {ep_idx}, '{audio_key}': MISSING")
                continue
            index_path = self.ds.root / DEFAULT_DATA_PATH.format(
                episode_chunk=ep_chunk, episode_index=ep_idx, feature_key=audio_key
            )
            try:
                import pyarrow.parquet as pq

                idx_table = pq.read_table(index_path)
                n_rows = len(idx_table)
                index_samples = int(
                    np.sum(idx_table.column("num_samples").to_numpy())
                )
                with wave.open(str(wav_path), "rb") as wf:
                    sample_rate = wf.getframerate()
                    channels = wf.getnchannels()
                    n_samples = wf.getnframes()
                duration_s = n_samples / sample_rate
                status = "OK" if n_samples == index_samples else (
                    f"sample count {n_samples} != index total {index_samples}"
                )
                print(f"[AUDIO] episode {ep_idx}, '{audio_key}': {status} "
                      f"({n_rows} chunks, {channels}ch, {sample_rate}Hz, "
                      f"{duration_s:.2f}s)")
            except Exception as e:
                print(f"[AUDIO] episode {ep_idx}, '{audio_key}': error — {e}")

    def _audio_keys(self) -> list[str]:
        return [
            key for key, ft in self.ds.meta.features.items()
            if ft.get("dtype") == "audio"
        ]

    def check_episode_alignment(
        self, ep_idx: int, episode_length: int, frame_records: list[dict[str, Any]]
    ) -> None:
        master_ts = np.array(
            [r["timestamp"] for r in frame_records], dtype=np.float64
        )
        report_path = self.ds.root / "meta" / "alignment_check.jsonl"

        for key in self.ds.meta.data_keys:
            ft = self.ds.meta.features.get(key, {})
            if ft.get("dtype") in ("video", "image"):
                continue
            tolerance = ft.get("tolerance_s", None)
            if tolerance is None:
                continue
            f = self.ds._features.get(key)
            if f is None:
                continue
            ts, _ = f.load(ep_idx)
            if len(ts) == 0:
                continue

            violations = []
            for frame_i in range(episode_length):
                mt = master_ts[frame_i]
                idx = np.argmin(np.abs(ts - mt))
                diff = float(abs(ts[idx] - mt))
                if diff > tolerance:
                    violations.append({
                        "frame": frame_i,
                        "t_master": round(float(mt), 6),
                        "t_nearest": round(float(ts[idx]), 6),
                        "diff": round(diff, 6),
                    })

            if violations:
                report = {
                    "episode_index": ep_idx, "feature": key,
                    "fps": ft.get("fps", None),
                    "tolerance_s": tolerance,
                    "total_frames": episode_length,
                    "total_readings": len(ts),
                    "violations": len(violations),
                    "max_diff": round(
                        float(max(abs(ts[np.argmin(np.abs(ts - mt))] - mt)
                                  for mt in master_ts)), 6),
                    "details": violations,
                }
                report_path.parent.mkdir(parents=True, exist_ok=True)
                append_jsonlines(report, report_path)
                print(
                    f"[ALIGN] episode {ep_idx}, feature '{key}': "
                    f"{len(violations)}/{episode_length} frames exceed "
                    f"tolerance_s={tolerance}:"
                )
                for v in violations[:5]:
                    print(f"  frame {v['frame']} (t={v['t_master']:.4f}) "
                          f"nearest at t={v['t_nearest']:.4f}, "
                          f"diff={v['diff']:.4f}s")
                if len(violations) > 5:
                    print(f"  ... ({len(violations) - 5} more)")
            else:
                print(f"[ALIGN] episode {ep_idx}, feature '{key}': "
                      f"OK ({episode_length} frames within tolerance_s={tolerance})")
