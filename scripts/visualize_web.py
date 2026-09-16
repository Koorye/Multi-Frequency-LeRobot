#!/usr/bin/env python
"""Web visualizer for mf_lerobot datasets.

Serves a browser UI (``web_visualizer.html``) plus a small JSON/JPEG API over
the dataset::

    python scripts/visualize_web.py [DATASET_ROOT] [--port 8000] [--no-browser]

DATASET_ROOT defaults to the newest directory under ``data/raw/``.

API::

    GET /api/dataset                     dataset + feature summary
    GET /api/frame?ep=&f=&w0=&w1=&cols=&filt=  windowed samples at frame f
    GET /api/video?ep=&f=&cam=&scale=&q= JPEG of one camera frame
    GET /api/stats?ep=                   per-feature min/max/mean/std (bands)

``filt`` (optional, per feature) applies zero-phase filtering to a signal
before windowing, e.g. ``filt=observation.eeg:hp=0.5,lp=40,notch=50`` —
``hp``/``lp`` are 4th-order Butterworth cutoffs in Hz, ``notch`` is a comb
notch at that line frequency and all its harmonics (50 → 50/100/150/… Hz).
Filtered full-episode signals are cached (bounded); responses carry the
applied spec under ``samples[key]["filter"]``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import threading
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pyarrow.parquet as pq
from scipy import signal as sp_signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mf_lerobot import MultiFrequencyLeRobotDataset  # noqa: E402
from mf_lerobot.utils import DEFAULT_VIDEO_PATH  # noqa: E402
from mf_lerobot.video import VideoFeature  # noqa: E402


def _sanitize(obj):
    """Replace non-finite floats (NaN/Inf) with None — sensor streams contain
    NaNs (lost glove tracking, invalid gaze...), and Python's json.dumps would
    emit bare ``NaN`` which is not valid JSON and breaks ``res.json()``."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    return obj

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = Path(__file__).resolve().parent / "web_visualizer.html"

DEFAULT_WINDOW = (-0.033, 0.0)  # (w0, w1] seconds relative to the frame
MAX_SAMPLES = 1500               # stride-decimate responses beyond this
VIDEO_CACHE = 160                # decoded camera JPEGs kept in memory
BURST = 8                        # frames decoded per cache miss (smaller for big frames)
FILTER_CACHE_BYTES = 600 << 20   # budget for cached filtered full-episode signals
FILTER_ORDER = 4                 # Butterworth order for hp/lp
NOTCH_Q = 30.0                   # comb-notch quality factor


def parse_filter_spec(spec: str) -> dict | None:
    """'hp=0.5,lp=40,notch=50' -> {'hp':0.5,'lp':40.0,'notch':50.0}; '' -> None."""
    out = {}
    for kv in spec.split(","):
        kv = kv.strip()
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            val = float(v)
        except ValueError:
            continue
        if k in ("hp", "lp", "notch") and 0 < val < 1e6:
            out[k] = val
    return out or None


def _line_extend(x: np.ndarray, pad: int) -> tuple[np.ndarray, int]:
    """Extend 1-D ``x`` by ``pad`` samples on both ends for filtering.

    Damped + clamped linear extrapolation: continuous in value and slope at
    the seam (no fold like odd mirroring), while the exponential decay and
    the robust-amplitude clamp stop a noisy edge slope from ramping the
    extension far outside the signal's real range — a runaway ramp would
    itself ring the high-pass at the edges.
    """
    n = len(x)
    pad = min(pad, n - 1)
    if pad <= 0:
        return x, 0
    w = min(n - 1, max(8, pad // 4))
    s0 = float(np.median(np.diff(x[:w])))       # robust signed edge slopes
    s1 = float(np.median(np.diff(x[n - w:])))
    lo, hi = np.nanpercentile(x, [0.5, 99.5])
    a_max = max(2.0 * (hi - lo), 1e-12)         # a few robust signal scales
    d = np.arange(1, pad + 1, dtype=np.float64)
    tau = max(pad / 3.0, 1.0)
    damp = d * np.exp(-d / tau)
    left = x[0] + np.clip(s0 * damp, -a_max, a_max)
    right = x[-1] + np.clip(s1 * damp, -a_max, a_max)
    return np.concatenate([left[::-1], x, right]), pad


def bandpass_filter(fts: np.ndarray, vals: np.ndarray, filt: dict,
                    fs_hint: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Zero-phase Butterworth hp/lp + comb notch (line freq + harmonics).

    Runs on the whole episode per channel, skipping NaN gaps (each finite
    segment is filtered on its own). The first/last ``margin`` samples of
    every segment are blanked afterwards: the filter's settling response to
    a segment edge (recording start, NaN gap) rings for a while and is not
    interpretable data. Returns a new array — the raw cache is never touched.
    """
    if vals.ndim != 2 or not len(fts):
        return fts, vals
    if fs_hint:
        fs = float(fs_hint)
    elif len(fts) > 1:
        dts = np.diff(fts)
        pos = dts[dts > 0]
        fs = 1.0 / float(np.median(pos)) if len(pos) else 0.0
    else:
        fs = 0.0
    if not fs or fs <= 0:
        raise ValueError("无法确定采样率，不能滤波")
    nyq = fs / 2.0

    sections = []
    hp, lp, notch = filt.get("hp"), filt.get("lp"), filt.get("notch")
    if hp:
        if not 0 < hp < nyq * 0.95:
            raise ValueError(f"高通 {hp} Hz 超出范围 (0, {nyq * 0.95:.1f})")
        sections.append(sp_signal.butter(FILTER_ORDER, hp, btype="highpass",
                                         fs=fs, output="sos"))
    if lp:
        if not 0 < lp < nyq * 0.95:
            raise ValueError(f"低通 {lp} Hz 超出范围 (0, {nyq * 0.95:.1f})")
        sections.append(sp_signal.butter(FILTER_ORDER, lp, btype="lowpass",
                                         fs=fs, output="sos"))
    if notch:
        for k in range(1, int(nyq * 0.95 / notch) + 1):
            b, a = sp_signal.iirnotch(k * notch, NOTCH_Q, fs=fs)
            sections.append(sp_signal.tf2sos(b, a))
    if not sections:
        return fts, vals
    sos = np.vstack(sections)
    # Edge padding must cover the slowest active cutoff's settling time, or
    # the filtfilt transient rings into the first/last seconds of the signal
    # (e.g. a 0.5 Hz high-pass needs seconds, not the default few dozen
    # samples). Capped to keep tiny episodes/segments workable.
    lowest_cut = min((v for v in (hp, lp, notch) if v), default=None)
    padlen = 3 * (2 * sos.shape[0]) + 1
    if lowest_cut:
        padlen = max(padlen, min(int(fs * 2.0 / lowest_cut), 10 * int(fs)))
    # Settle time of each active filter: the slow high-pass swing needs
    # ~1/cutoff, the comb-notch ring decays as exp(-pi*f0*t/Q) (~0.6 s to
    # a few percent for a 50 Hz, Q=30 notch). Segment edges are blanked
    # for this long.
    margin = 0
    if hp:
        margin = max(margin, int(fs * 1.0 / hp))
    if notch:
        margin = max(margin, int(fs * 0.6))

    out = vals.astype(np.float32, copy=True)
    inner = 3 * (2 * sos.shape[0]) + 1   # small odd pad on top of the line extension
    for j in range(out.shape[1]):
        col = out[:, j]
        good = np.flatnonzero(np.isfinite(col))
        if not len(good):
            continue
        breaks = np.flatnonzero(np.diff(good) > 1)
        starts = np.r_[good[0], good[breaks + 1]]
        ends = np.r_[good[breaks], good[-1] + 1]
        for s, e in zip(starts, ends):
            if e - s < 4:
                continue
            seg = col[s:e].astype(np.float64)
            ext, pad = _line_extend(seg, min(padlen, len(seg) // 2))
            col[s:e] = sp_signal.sosfiltfilt(
                sos, ext, padlen=min(inner, len(ext) - 1))[pad:pad + len(seg)]
            m = min(margin, (e - s) // 3)
            if m > 0:
                col[s:s + m] = np.nan
                col[e - m:e] = np.nan
    return fts, out


class DatasetApp:
    """Read-only view over one MultiFrequencyLeRobotDataset + caches."""

    def __init__(self, root: Path):
        self.root = root
        self.ds = MultiFrequencyLeRobotDataset(root.name, root=root)
        self.chunks_size = int(self.ds.meta.info.get("chunks_size", 1000))
        self._ep_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._video_cache: "OrderedDict[tuple, bytes]" = OrderedDict()
        self._filt_cache: "OrderedDict[tuple, tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self._filt_bytes = 0
        self._lock = threading.Lock()

    # ── Episode master tables ──

    def episode_table(self, ep: int) -> tuple[np.ndarray, np.ndarray]:
        """(timestamps float64, task_index int64) of one episode's master frames."""
        if ep not in self._ep_cache:
            chunk = ep // self.chunks_size
            path = self.root / f"data/chunk-{chunk:03d}/episode_{ep:06d}/master_index.parquet"
            table = pq.read_table(path)
            ts = table.column("timestamp").to_numpy().astype(np.float64)
            task_idx = table.column("task_index").to_numpy()
            self._ep_cache[ep] = (ts, task_idx)
        return self._ep_cache[ep]

    def task_name(self, task_index: int) -> str:
        return self.ds.meta.tasks.get(int(task_index), "")

    def filtered_episode(self, ep: int, key: str, filt: dict) -> tuple[np.ndarray, np.ndarray]:
        """Full-episode (timestamps, filtered values), cached per (ep, key, params)."""
        ck = (ep, key, tuple(sorted(filt.items())))
        with self._lock:
            if ck in self._filt_cache:
                self._filt_cache.move_to_end(ck)
                return self._filt_cache[ck]
        feat = self.ds._features.get(key)
        fs_hint = float(feat.spec.get("fps") or 0) if feat is not None else None
        fts, vals = feat.load(ep)
        fts, vals = bandpass_filter(fts, vals, filt, fs_hint)
        with self._lock:
            if ck not in self._filt_cache:
                self._filt_cache[ck] = (fts, vals)
                self._filt_bytes += vals.nbytes + fts.nbytes
                while self._filt_bytes > FILTER_CACHE_BYTES and len(self._filt_cache) > 1:
                    _, (t2, v2) = self._filt_cache.popitem(last=False)
                    self._filt_bytes -= v2.nbytes + t2.nbytes
            return self._filt_cache[ck]

    # ── Data queries ──

    def frame_data(self, ep: int, f: int, w0: float, w1: float,
                   cols: dict[str, list[int] | None],
                   filt: dict[str, dict] | None = None) -> dict:
        ts_ep, task_idx = self.episode_table(ep)
        if f < 0 or f >= len(ts_ep):
            raise ValueError(f"frame {f} out of range for episode {ep} (0..{len(ts_ep) - 1})")
        t = float(ts_ep[f])

        samples: dict[str, dict] = {}
        for key, feat in self.ds._features.items():
            if isinstance(feat, VideoFeature):
                continue
            flt = (filt or {}).get(key)
            if flt:
                fts, vals = self.filtered_episode(ep, key, flt)
            else:
                fts, vals = feat.load(ep)
            if w0 == w1:
                # "nearest" mode: the single sample closest to the frame
                i = int(np.argmin(np.abs(fts - t)))
                sel_ts, sel_vals = fts[i:i + 1], vals[i:i + 1]
            else:
                mask = (fts > t + w0) & (fts <= t + w1)
                sel_ts, sel_vals = fts[mask], vals[mask]
            if not len(sel_ts):
                continue
            sub = cols.get(key) if cols else None
            if sub is not None:
                sel_vals = sel_vals[:, sub]
            # stride-decimate big windows so the payload stays snappy
            if len(sel_ts) > MAX_SAMPLES:
                k = int(np.ceil(len(sel_ts) / MAX_SAMPLES))
                sel_ts, sel_vals = sel_ts[::k], sel_vals[::k]
            spec = self.ds.meta.features.get(key, {})
            names = spec.get("names") or [f"dim_{i}" for i in range(vals.shape[1])]
            if sub is not None:
                names = [names[i] for i in sub]
            samples[key] = {
                "ts": (sel_ts - t).round(9).tolist(),   # relative to the frame
                "values": sel_vals.tolist(),
                "names": names,
                # original column indices of the returned subset (channel
                # colors/filters stay stable under cols= filtering)
                "indices": sub if sub is not None else list(range(vals.shape[1])),
            }
            if flt:
                samples[key]["filter"] = flt

        return _sanitize({
            "episode": ep,
            "frame_index": f,
            "timestamp": t,
            "task": self.task_name(task_idx[f]),
            "samples": samples,
        })

    def video_jpeg(self, ep: int, f: int, cam: str, scale: float, quality: int) -> bytes:
        feat = self.ds._features.get(cam)
        if not isinstance(feat, VideoFeature):
            raise ValueError(f"'{cam}' is not a video feature")
        ts_ep, _ = self.episode_table(ep)
        ts_v = feat._load_timestamps(ep)
        if not len(ts_v):
            raise ValueError(f"video '{cam}' has no frames")
        idx = int(np.argmin(np.abs(ts_v - float(ts_ep[f]))))

        def key(k: int):
            return (ep, k, cam, scale, quality)

        with self._lock:
            if key(idx) in self._video_cache:
                self._video_cache.move_to_end(key(idx))
                return self._video_cache[key(idx)]

        # Cache miss: decode a burst of frames around idx in one open+seek,
        # so sequential scrubbing/playback hits the cache.
        import av
        from PIL import Image

        fps = float(feat.spec.get("fps", 30) or 30)
        h, w = feat.spec.get("shape", (0, 0, 3))[:2]
        radius = 3 if h * w > 800_000 else BURST
        lo, hi = max(0, idx - radius), min(len(ts_v) - 1, idx + radius)

        video_path = self.root / DEFAULT_VIDEO_PATH.format(
            episode_chunk=ep // self.chunks_size, video_key=cam, episode_index=ep)
        burst: dict[int, bytes] = {}
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            try:
                container.seek(int(lo / fps / stream.time_base), stream=stream)
            except Exception:
                pass  # decode from the start; frames before lo are skipped below
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                k = int(round(frame.pts * float(frame.time_base) * fps))
                if k < lo:
                    continue
                if k > hi:
                    break
                arr = frame.to_ndarray(format="rgb24")
                pil = Image.fromarray(arr)
                if scale != 1.0:
                    pil = pil.resize(
                        (max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
                        Image.BILINEAR,
                    )
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=quality)
                burst[k] = buf.getvalue()

        with self._lock:
            for k, data in burst.items():
                self._video_cache[key(k)] = data
            while len(self._video_cache) > VIDEO_CACHE:
                self._video_cache.popitem(last=False)
            if key(idx) in self._video_cache:
                return self._video_cache[key(idx)]
        raise ValueError(f"video '{cam}' frame {idx} not decodable")

    # ── Metadata payloads ──

    def dataset_summary(self) -> dict:
        features = []
        for key, ft in self.ds.meta.features.items():
            kind = "video" if ft.get("dtype") in ("video", "image") else "parquet"
            if key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
                continue
            shape = list(ft.get("shape", []))
            names = ft.get("names")
            if names is None and shape:
                names = [f"dim_{i}" for i in range(shape[0])]
            features.append({
                "key": key,
                "kind": kind,
                "shape": shape,
                "names": names,
                "window": list(ft["window"]) if ft.get("window") else None,
                "fps": ft.get("fps"),
            })
        return {
            "name": self.ds.repo_id,
            "fps": float(self.ds.meta.fps or 30),
            "master_feature": self.ds.meta.master_feature,
            "episodes": [
                {
                    "index": ep,
                    "length": info["length"],
                    "tasks": info["tasks"],
                }
                for ep, info in sorted(self.ds.meta.episodes.items())
            ],
            "features": features,
        }

    def episode_stats(self, ep: int) -> dict:
        out = {}
        for key, st in self.ds.meta.episodes_stats.get(ep, {}).items():
            out[key] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in st.items() if k in ("min", "max", "mean", "std")
            }
        return _sanitize(out)


# ── HTTP ──────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    app: DatasetApp = None  # injected at server build time

    def log_message(self, fmt, *args):  # quieter than the default
        sys.stderr.write("[viz] %s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path == "/" or url.path == "/index.html":
                self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/dataset":
                self._json(self.app.dataset_summary())
            elif url.path == "/api/frame":
                ep = int(q["ep"][0]); f = int(q["f"][0])
                w0 = float(q.get("w0", [DEFAULT_WINDOW[0]])[0])
                w1 = float(q.get("w1", [DEFAULT_WINDOW[1]])[0])
                cols = {}
                for part in q.get("cols", [])[0].split(";") if q.get("cols") else []:
                    if ":" not in part:
                        continue
                    key, spec = part.split(":", 1)
                    cols[key] = (None if spec == "*" else
                                 [] if spec == "" else
                                 [int(x) for x in spec.split(",")])
                filt = {}
                for part in q.get("filt", [])[0].split(";") if q.get("filt") else []:
                    if ":" not in part:
                        continue
                    key, spec = part.split(":", 1)
                    fspec = parse_filter_spec(spec)
                    if fspec:
                        filt[key] = fspec
                self._json(self.app.frame_data(ep, f, w0, w1, cols, filt))
            elif url.path == "/api/video":
                ep = int(q["ep"][0]); f = int(q["f"][0]); cam = q["cam"][0]
                scale = float(q.get("scale", ["1.0"])[0])
                quality = int(q.get("q", ["80"])[0])
                self._send(200, self.app.video_jpeg(ep, f, cam, scale, quality),
                           "image/jpeg")
            elif url.path == "/api/stats":
                self._json(self.app.episode_stats(int(q["ep"][0])))
            else:
                self._json({"error": f"unknown path {url.path}"}, 404)
        except KeyError as e:
            self._json({"error": f"missing query parameter {e}"}, 400)
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:  # surface anything else as JSON, keep serving
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def default_root() -> Path:
    raw = PROJECT_ROOT / "data" / "raw"
    if raw.is_dir():
        dirs = sorted([p for p in raw.iterdir() if p.is_dir()],
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if dirs:
            return dirs[0]
    raise SystemExit(f"[error] 未指定数据集目录，且 {raw} 下没有数据集")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", nargs="?", type=Path, default=None,
                    help="数据集目录 (default: data/raw 下最新的)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    root = args.root or default_root()
    if not (root / "meta" / "info.json").exists():
        raise SystemExit(f"[error] '{root}' 不是 mf_lerobot 数据集 (缺 meta/info.json)")
    if not HTML_PATH.exists():
        raise SystemExit(f"[error] 找不到前端页面 {HTML_PATH}")

    print(f"[viz] loading {root} ...")
    app = DatasetApp(root)
    summary = app.dataset_summary()
    print(f"[viz] {summary['name']}: {len(summary['episodes'])} episodes, "
          f"{len(summary['features'])} features")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    Handler.app = app
    url = f"http://{args.host}:{args.port}/"
    print(f"[viz] serving {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[viz] bye")


if __name__ == "__main__":
    main()
