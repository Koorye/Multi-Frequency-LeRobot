"""Constants for the LeRobot v3.1 multi-frequency extension."""

CODEBASE_VERSION = "v3.1"

# ── Path formats ──

# Per-feature data path: every non-video feature gets its own parquet
DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}/{feature_key}.parquet"

# Frame index parquet: timestamp, frame_index, episode_index, index, task_index, task
DEFAULT_INDEX_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}/master_index.parquet"

# Video path
DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

# Info / stats paths
INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
DEFAULT_TASKS_PATH = "meta/tasks.parquet"


# ── Episode metadata key helpers for per-feature data ──

def feature_chunk_key(feature_key: str) -> str:
    """Episode metadata key for a feature's chunk index."""
    return f"data/{feature_key}/chunk_index"


def feature_file_key(feature_key: str) -> str:
    """Episode metadata key for a feature's file index."""
    return f"data/{feature_key}/file_index"


def feature_from_ts_key(feature_key: str) -> str:
    """Episode metadata key for a feature's start timestamp."""
    return f"data/{feature_key}/from_timestamp"


def feature_to_ts_key(feature_key: str) -> str:
    """Episode metadata key for a feature's end timestamp."""
    return f"data/{feature_key}/to_timestamp"


def feature_length_key(feature_key: str) -> str:
    """Episode metadata key for a feature's sample count."""
    return f"data/{feature_key}/length"


# ── Feature defaults ──

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float64", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}



import json
from pathlib import Path
from typing import Any

# ── All constants defined above; no external imports needed ──


def create_empty_dataset_info(
    fps: float,
    features: dict[str, dict],
    master_feature: str | None = None,
    use_videos: bool = True,
    robot_type: str | None = None,
) -> dict[str, Any]:
    """Create the info.json template for a new v3.1 multi-frequency dataset.

    All non-video features are stored as individual parquet files under
    data/chunk-{c}/episode_{idx}_{feature_key}.parquet. A lightweight
    frame index parquet tracks per-frame metadata.

    Args:
        fps: Approximate frames per second of the master camera (informational).
        features: Full feature dictionary (including DEFAULT_FEATURES).
        master_feature: Key of the camera feature that defines the master clock.
        use_videos: Whether visual features are stored as videos.
        robot_type: Optional robot type string.
        chunks_size: Max episodes per chunk directory.
        data_files_size_in_mb: Max size of data parquet files.
        video_files_size_in_mb: Max size of video files.

    Returns:
        dict ready to be written as info.json.
    """
    # Validate master_feature
    camera_keys = _get_camera_keys(features)
    if master_feature is None and len(camera_keys) > 0:
        master_feature = camera_keys[0]
    if master_feature is not None and master_feature not in camera_keys:
        raise ValueError(
            f"master_feature '{master_feature}' not found in camera features: {camera_keys}"
        )

    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "master_feature": master_feature,
        "fps": fps,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "splits": {},
        "data_path": DEFAULT_DATA_PATH,
        "index_path": DEFAULT_INDEX_PATH,
        "video_path": DEFAULT_VIDEO_PATH if use_videos else None,
        "features": features,
    }


def _get_camera_keys(features: dict[str, dict]) -> list[str]:
    """Get all video/image feature keys."""
    return [
        key for key, ft in features.items()
        if ft.get("dtype") in ("video", "image")
    ]


def classify_features(
    features: dict[str, dict],
    master_feature: str | None = None,
) -> dict[str, list[str]]:
    """Classify feature keys into groups.

    Returns:
        Dict with keys: 'default' (timestamp, index etc), 'data'
        (non-video sensor features), 'camera' (all video/image).
    """
    default_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index"}

    camera_keys: set[str] = set()
    data_keys: set[str] = set()

    for key, ft in features.items():
        if key in default_keys:
            continue
        if ft.get("dtype") in ("video", "image"):
            camera_keys.add(key)
        else:
            data_keys.add(key)

    return {
        "default": sorted(default_keys & set(features.keys())),
        "data": sorted(data_keys),
        "camera": sorted(camera_keys),
    }


def write_info(info: dict, root: Path) -> None:
    """Write info.json to the dataset root."""
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info_path = meta_dir / "info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)


def load_info(root: Path) -> dict:
    """Load info.json from the dataset root."""
    info_path = root / INFO_PATH
    with open(info_path) as f:
        info = json.load(f)

    # Convert shapes to tuples for consistency
    for ft in info.get("features", {}).values():
        if "shape" in ft:
            ft["shape"] = tuple(ft["shape"])

    return info


def validate_features(features: dict[str, dict]) -> None:
    """Validate feature specifications.

    Raises ValueError for invalid feature definitions.
    """
    for key, ft in features.items():
        # Check required fields
        if "dtype" not in ft:
            raise ValueError(f"Feature '{key}' missing required field 'dtype'")
        if "shape" not in ft:
            raise ValueError(f"Feature '{key}' missing required field 'shape'")
        if "names" not in ft:
            raise ValueError(f"Feature '{key}' missing required field 'names'")

        # No '/' in feature names (LeRobot convention)
        if "/" in key:
            raise ValueError(
                f"Feature names should not contain '/'. Found in '{key}'."
            )

        # Validate window if present
        if "window" in ft:
            w = ft["window"]
            if w is not None and w != "interpolate":
                if not (isinstance(w, (list, tuple)) and len(w) == 2
                        and all(isinstance(x, (int, float)) for x in w)):
                    raise ValueError(
                        f"Invalid window '{w}' for feature '{key}'. "
                        f"Must be None, 'interpolate', or (start_s, end_s) tuple."
                    )
