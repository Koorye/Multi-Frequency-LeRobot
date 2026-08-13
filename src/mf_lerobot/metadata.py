"""Multi-frequency dataset metadata — mirrors LeRobotDatasetMetadata."""

from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from .utils import (
    DEFAULT_FEATURES,
    create_empty_dataset_info as create_v31_info,
    validate_features,
    write_info,
)


class MultiFrequencyDatasetMetadata(LeRobotDatasetMetadata):
    """Extended metadata for multi-frequency datasets."""

    def load_metadata(self):
        super().load_metadata()

        self._master_feature = self.info.get("master_feature", None)
        self._data_keys: list[str] = []
        self._feature_windows: dict[str, tuple | str | None] = {}
        self._feature_fps_map: dict[str, float | None] = {}

        for key, ft in self.features.items():
            dtype = ft.get("dtype", "")
            if dtype in ("video", "image"):
                continue
            if key in DEFAULT_FEATURES:
                continue
            self._data_keys.append(key)
            window = ft.get("window", None)
            if window is not None:
                self._feature_windows[key] = window
            feature_fps = ft.get("fps")
            if feature_fps is not None:
                self._feature_fps_map[key] = feature_fps

    @property
    def master_feature(self) -> str | None:
        return self._master_feature

    @property
    def data_keys(self) -> list[str]:
        return self._data_keys

    @property
    def feature_windows(self) -> dict[str, tuple | str | None]:
        return self._feature_windows

    def get_window(self, key: str) -> tuple | str | None:
        return self._feature_windows.get(key, None)

    def get_feature_fps(self, key: str) -> float | None:
        return self._feature_fps_map.get(key, None)

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: float,
        features: dict,
        master_feature: str | None = None,
        robot_type: str | None = None,
        root: str | Path | None = None,
        use_videos: bool = True,
    ) -> "MultiFrequencyDatasetMetadata":
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = (
            Path(root)
            if root is not None
            else Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
        )
        obj.root.mkdir(parents=True, exist_ok=False)

        all_features = {**features, **DEFAULT_FEATURES}
        validate_features(all_features)

        camera_keys = [
            k for k, ft in all_features.items()
            if ft.get("dtype") in ("video", "image")
        ]
        if master_feature is None and len(camera_keys) > 0:
            master_feature = camera_keys[0]

        obj._master_feature = master_feature
        obj._data_keys = []
        obj._feature_windows = {}
        obj._feature_fps_map = {}
        for key, ft in all_features.items():
            dtype = ft.get("dtype", "")
            if dtype in ("video", "image"):
                continue
            if key in DEFAULT_FEATURES:
                continue
            obj._data_keys.append(key)
            window = ft.get("window")
            if window is not None:
                obj._feature_windows[key] = window
            feature_fps = ft.get("fps")
            if feature_fps is not None:
                obj._feature_fps_map[key] = feature_fps

        obj.info = create_v31_info(
            fps=fps, features=all_features,
            master_feature=master_feature,
            use_videos=use_videos,
            robot_type=robot_type,
        )
        write_info(obj.info, obj.root)
        obj.revision = None
        obj.tasks = {}
        obj.task_to_task_index = {}
        obj.episodes = {}
        obj.stats = None
        obj.episodes_stats = {}
        obj.writer = None
        obj.latest_episode = None
        obj.metadata_buffer = []
        obj.metadata_buffer_size = 10
        return obj
