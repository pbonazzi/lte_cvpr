from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize

from data.transforms.event_augmentations import from_tonic_structured

CIFAR10_DVS_DIRNAME = "CIFAR10-DVS"
# Use the ndownloader host directly: https://figshare.com/ndownloader/files/<id>
# now answers HTTP 202 with an empty body indefinitely instead of redirecting.
CIFAR10_DVS_DOWNLOAD_URL = "https://ndownloader.figshare.com/files/38023437"
CIFAR10_DVS_ARCHIVE_NAME = "CIFAR10DVS.zip"
CIFAR10_DVS_ARCHIVE_MD5 = "ce3a4a0682dc0943703bd8f749a7701c"
CIFAR10_DVS_CLASS_ARCHIVES = [
    "airplane.zip",
    "automobile.zip",
    "bird.zip",
    "cat.zip",
    "deer.zip",
    "dog.zip",
    "frog.zip",
    "horse.zip",
    "ship.zip",
    "truck.zip",
]
CIFAR10_DVS_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
CIFAR10_DVS_SENSOR_SIZE = (128, 128)


def _require_tonic():
    try:
        import tonic
    except ImportError as exc:
        raise ImportError(
            "tonic is required for CIFAR10-DVS support. Install it first, for example via "
            "`pip install tonic` or by recreating the project environment."
        ) from exc
    return tonic


def _require_tonic_download_utils():
    _require_tonic()
    import tonic.download_utils as download_utils

    return download_utils


def resolve_cifar10_dvs_root(data_path: Path | str) -> Path:
    root = Path(data_path)
    if root.name == CIFAR10_DVS_DIRNAME:
        return root
    return root / CIFAR10_DVS_DIRNAME


def _candidate_dataset_roots(data_path: Path | str):
    root = resolve_cifar10_dvs_root(data_path)
    # Support both the intended folder layout and the nested layout produced by
    # tonic.datasets.CIFAR10DVS(save_to=...).
    return (root, root / "CIFAR10DVS")


def _count_aedat4_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*.aedat4"))


def _find_existing_dataset_root(data_path: Path | str):
    for candidate in _candidate_dataset_roots(data_path):
        if _count_aedat4_files(candidate) > 0:
            return candidate
    return None


def download_cifar10_dvs(data_path: Path | str) -> Path:
    existing_root = _find_existing_dataset_root(data_path)
    if existing_root is not None:
        return existing_root

    root = resolve_cifar10_dvs_root(data_path)
    root.mkdir(parents=True, exist_ok=True)

    download_utils = _require_tonic_download_utils()
    archive_path = root / CIFAR10_DVS_ARCHIVE_NAME
    try:
        download_utils.download_and_extract_archive(
            CIFAR10_DVS_DOWNLOAD_URL,
            download_root=str(root),
            extract_root=str(root),
            filename=CIFAR10_DVS_ARCHIVE_NAME,
            md5=CIFAR10_DVS_ARCHIVE_MD5,
        )
        for class_archive in CIFAR10_DVS_CLASS_ARCHIVES:
            class_archive_path = root / class_archive
            if class_archive_path.exists():
                download_utils.extract_archive(str(class_archive_path), str(root))
    except Exception as exc:
        raise RuntimeError(
            "Failed to download CIFAR10-DVS automatically. "
            f"Tried to populate {root}. "
            f"Tonic reported: {exc}. "
            f"If you download the dataset manually, extract the class folders directly under {root}. "
            f"If a partial archive exists at {archive_path}, delete it and retry."
        ) from exc

    existing_root = _find_existing_dataset_root(root)
    if existing_root is None:
        raise FileNotFoundError(
            f"CIFAR10-DVS download finished but no .aedat4 files were found under {root}"
        )
    return existing_root


@lru_cache(maxsize=None)
def _load_cifar10_dvs_metadata_cached(root_path: str):
    root = download_cifar10_dvs(root_path)

    class_to_idx = {name: idx for idx, name in enumerate(CIFAR10_DVS_CLASS_NAMES)}
    file_paths = []
    labels = []

    for class_name in CIFAR10_DVS_CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.exists():
            continue

        for file_path in sorted(class_dir.rglob("*.aedat4")):
            file_paths.append(str(file_path))
            labels.append(class_to_idx[class_name])

    if not file_paths:
        raise FileNotFoundError(f"No CIFAR10-DVS .aedat4 files found under {root}")

    return tuple(file_paths), np.asarray(labels, dtype=np.int64)


def load_cifar10_dvs_metadata(data_path: Path | str):
    root = resolve_cifar10_dvs_root(data_path).resolve()
    file_paths, labels = _load_cifar10_dvs_metadata_cached(str(root))
    return list(file_paths), labels.copy()


def build_cifar10_dvs_splits(
    data_path: Path | str,
    train_size: float = 0.9,
    seed: int = 15,
    val_ratio_within_train: float = 0.1,
):
    if not 0.0 < train_size < 1.0:
        raise ValueError(f"train_size must be in (0, 1), got {train_size}")
    if not 0.0 <= val_ratio_within_train < 1.0:
        raise ValueError(
            f"val_ratio_within_train must be in [0, 1), got {val_ratio_within_train}"
        )

    _, labels = load_cifar10_dvs_metadata(data_path)
    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []
    test_indices = []

    for class_idx in range(len(CIFAR10_DVS_CLASS_NAMES)):
        class_indices = np.flatnonzero(labels == class_idx)
        class_indices = rng.permutation(class_indices)

        trainval_end = int(len(class_indices) * train_size)
        trainval_indices = class_indices[:trainval_end]
        test_indices.extend(class_indices[trainval_end:].tolist())

        val_count = int(len(trainval_indices) * val_ratio_within_train)
        val_indices.extend(trainval_indices[:val_count].tolist())
        train_indices.extend(trainval_indices[val_count:].tolist())

    return {
        "train": rng.permutation(np.asarray(train_indices, dtype=np.int64)),
        "val": rng.permutation(np.asarray(val_indices, dtype=np.int64)),
        "test": rng.permutation(np.asarray(test_indices, dtype=np.int64)),
    }


class CIFAR10DVS(Dataset):
    def __init__(
        self,
        data_path: Path | str,
        indices=None,
        transform=None,
        representation: str = "spike_tensor",
        target_size: Sequence[int] = (32, 32),
        num_time_bins: int = 9,
        event_filter=None,
        event_transform=None,
        binning_strategy: str = "duration",
        use_cache: bool = False,
        denoise_filter_time_us: float = None,
        canonicalize_orientation: bool = False,
    ):
        self.transform = transform
        self.representation = representation
        self.target_size = tuple(int(dim) for dim in target_size)
        self.num_time_bins = int(num_time_bins)
        self.event_filter = event_filter
        self.event_transform = event_transform
        self.binning_strategy = str(binning_strategy)
        self.sensor_height, self.sensor_width = CIFAR10_DVS_SENSOR_SIZE
        self.resize = Resize(self.target_size, interpolation=InterpolationMode.NEAREST)
        
        # Cache settings
        self.use_cache = use_cache and representation == "spike_tensor"
        self.denoise_filter_time_us = denoise_filter_time_us
        # Sibling of the CIFAR10-DVS dataset folder, matching where
        # cifar10_dvs_preprocess.create_cache_directory() writes the cache.
        self.cache_dir = Path(data_path) / "CIFAR10-DVS-cache" if self.use_cache else None
        self.cache_config_tag = self._build_cache_config_tag()
        # This is the empirically better CIFAR10-DVS orientation for both
        # visualization and model training.
        self.canonicalize_orientation = bool(canonicalize_orientation)
        
        # Denoiser for fallback if cache doesn't exist
        if self.denoise_filter_time_us is not None:
            from data.transforms import Denoise
            self.denoiser = Denoise(filter_time=self.denoise_filter_time_us)
        else:
            self.denoiser = event_filter

        # For spike tensors, apply supported train-time augmentations after the
        # tensor has been resized and canonicalized so flip/jitter semantics are
        # aligned with the final orientation the model sees.
        self.tensor_transform = (
            self._build_tensor_transform(event_transform)
            if self.representation == "spike_tensor"
            else None
        )

        if self.representation not in {"histogram", "binary", "spike_tensor"}:
            raise ValueError(f"Unsupported representation: {self.representation}")
        if self.num_time_bins <= 0:
            raise ValueError(f"num_time_bins must be positive, got {self.num_time_bins}")
        if self.binning_strategy not in {"duration", "event_count"}:
            raise ValueError(f"Unsupported binning_strategy: {self.binning_strategy}")

        all_file_paths, all_labels = load_cifar10_dvs_metadata(data_path)
        if indices is None:
            indices = np.arange(len(all_labels), dtype=np.int64)
        else:
            indices = np.asarray(indices, dtype=np.int64)

        self.file_paths = [all_file_paths[idx] for idx in indices]
        self.labels = torch.as_tensor(all_labels[indices], dtype=torch.long)
        self.all_file_paths = all_file_paths
        
        # Build class-to-paths mapping for cache filename generation
        self._build_class_path_mapping(all_file_paths)

    def _build_tensor_transform(self, event_transform):
        """Convert event-level augmentations to tensor-level augmentations."""
        if event_transform is None:
            return None
        
        from data.transforms import (
            EventTransformCompose,
            SpatialJitter,
            RandomFlipLR,
            TensorEventTransformCompose,
            TensorSpatialJitter,
            TensorRandomFlipLR,
        )
        
        # Only process EventTransformCompose
        if not isinstance(event_transform, EventTransformCompose):
            return None
        
        tensor_transforms = []
        for transform in event_transform.transforms:
            if isinstance(transform, SpatialJitter):
                tensor_transforms.append(TensorSpatialJitter(
                    max_shift=transform.max_shift,
                    sensor_size=transform.sensor_size
                ))
            elif isinstance(transform, RandomFlipLR):
                tensor_transforms.append(TensorRandomFlipLR(p=transform.p))
            # Other transforms are skipped for tensor-level augmentation
        
        if not tensor_transforms:
            return None
        
        return TensorEventTransformCompose(tensor_transforms)

    def __len__(self):
        return len(self.labels)

    def _build_class_path_mapping(self, all_file_paths):
        """Build mapping from file_path to (class_name, instance_idx) for cache filenames."""
        self.file_path_to_cache_info = {}
        class_counters = {name: [] for name in CIFAR10_DVS_CLASS_NAMES}
        
        for idx, file_path in enumerate(all_file_paths):
            for class_name in CIFAR10_DVS_CLASS_NAMES:
                if f"/{class_name}/" in file_path:
                    class_counters[class_name].append((idx, file_path))
                    break
        
        for class_name, paths_list in class_counters.items():
            for instance_idx, (global_idx, file_path) in enumerate(paths_list):
                self.file_path_to_cache_info[file_path] = (class_name, instance_idx)

    @staticmethod
    def _format_cache_float(value) -> str:
        if value is None:
            return "none"
        return f"{float(value):g}".replace("-", "m").replace(".", "p")

    def _build_cache_config_tag(self) -> str:
        denoise_tag = self._format_cache_float(self.denoise_filter_time_us)
        return f"tb{self.num_time_bins}_{self.binning_strategy}_denoise{denoise_tag}"
    
    def _get_cache_filename(self, file_path: str) -> str:
        """Get cache filename for a given file_path."""
        if file_path not in self.file_path_to_cache_info:
            return None
        class_name, instance_idx = self.file_path_to_cache_info[file_path]
        return f"{class_name}_{instance_idx}_{self.cache_config_tag}.pt"
    
    def _get_cache_path(self, file_path: str) -> Path:
        """Get full cache path for a given file_path."""
        if not self.use_cache or self.cache_dir is None:
            return None
        cache_filename = self._get_cache_filename(file_path)
        if cache_filename is None:
            return None
        return self.cache_dir / cache_filename

    def __len__(self):
        return len(self.labels)

    def _read_events(self, file_path: str) -> np.ndarray:
        tonic = _require_tonic()
        events = tonic.io.read_aedat4(file_path)
        # tonic.io.read_aedat4 already returns structured events with semantic
        # field names ("x", "y", "t", "p"). Renaming them corrupts the event
        # stream by treating timestamps as coordinates.
        return from_tonic_structured(events).astype(np.float32, copy=False)

    @staticmethod
    def _sort_events_by_time(events: np.ndarray) -> np.ndarray:
        if len(events) > 1 and np.any(events[1:, 2] < events[:-1, 2]):
            order = np.argsort(events[:, 2], kind="mergesort")
            events = events[order]
        return events

    def _compute_bin_ids(self, events: np.ndarray) -> np.ndarray:
        # Map each timestamp to a discrete bin within this sample's time range.
        if len(events) == 0:
            return np.zeros(0, dtype=np.int64)

        if self.binning_strategy == "event_count":
            event_indices = np.arange(len(events), dtype=np.int64)
            bin_ids = np.floor(event_indices * self.num_time_bins / len(events)).astype(np.int64)
            return np.clip(bin_ids, 0, self.num_time_bins - 1)

        timestamps = events[:, 2]
        first_ts = float(timestamps[0])
        last_ts = float(timestamps[-1])
        if last_ts <= first_ts:
            return np.zeros(len(events), dtype=np.int64)

        relative = (timestamps - first_ts) / (last_ts - first_ts)
        bin_ids = np.floor(relative * self.num_time_bins).astype(np.int64)
        return np.clip(bin_ids, 0, self.num_time_bins - 1)

    def _events_to_frame(self, events: np.ndarray) -> torch.Tensor:
        frame = np.zeros((2, self.sensor_height, self.sensor_width), dtype=np.float32)
        if len(events) == 0:
            return torch.from_numpy(frame)

        x = np.clip(events[:, 0].astype(np.int32), 0, self.sensor_width - 1)
        y = np.clip(events[:, 1].astype(np.int32), 0, self.sensor_height - 1)
        pol = events[:, 3].astype(np.int32)

        pos_mask = pol == 1
        neg_mask = pol == 0

        if self.representation == "histogram":
            np.add.at(frame[0], (y[pos_mask], x[pos_mask]), 1.0)
            np.add.at(frame[1], (y[neg_mask], x[neg_mask]), 1.0)
        else:
            frame[0][y[pos_mask], x[pos_mask]] = 1.0
            frame[1][y[neg_mask], x[neg_mask]] = 1.0

        return torch.from_numpy(frame)

    def _events_to_spike_tensor(self, events: np.ndarray) -> torch.Tensor:
        # Build one binary occupancy map per time bin and polarity.
        spike = np.zeros(
            (self.num_time_bins, 2, self.sensor_height, self.sensor_width),
            dtype=np.float32,
        )
        if len(events) == 0:
            return torch.from_numpy(spike.reshape(2 * self.num_time_bins, self.sensor_height, self.sensor_width))

        events = self._sort_events_by_time(np.asarray(events, dtype=np.float32))
        bin_ids = self._compute_bin_ids(events)
        x = np.clip(events[:, 0].astype(np.int32), 0, self.sensor_width - 1)
        y = np.clip(events[:, 1].astype(np.int32), 0, self.sensor_height - 1)
        pol = events[:, 3].astype(np.int32)

        for bin_idx in range(self.num_time_bins):
            in_bin = bin_ids == bin_idx
            if not np.any(in_bin):
                continue

            pos_mask = in_bin & (pol == 1)
            neg_mask = in_bin & (pol == 0)
            # Multiple events at the same bin/polarity/pixel are collapsed to 1
            # because the model consumes presence/absence rather than counts.
            spike[bin_idx, 0, y[pos_mask], x[pos_mask]] = 1.0
            spike[bin_idx, 1, y[neg_mask], x[neg_mask]] = 1.0

        # Merge temporal bins and polarity into channels: [2B, H, W].
        return torch.from_numpy(spike.reshape(2 * self.num_time_bins, self.sensor_height, self.sensor_width))

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        
        # Try to load from cache if enabled
        if self.use_cache:
            cache_path = self._get_cache_path(file_path)
            if cache_path is not None and cache_path.exists():
                frame = torch.load(cache_path, weights_only=True)
            else:
                # Cache miss: generate from events
                events = self._read_events(file_path)
                
                # Apply denoise (either from denoise_filter_time_us or event_filter)
                if self.denoiser is not None:
                    events = self.denoiser(events, sensor_size=CIFAR10_DVS_SENSOR_SIZE)
                
                # Generate spike tensor
                frame = self._events_to_spike_tensor(events)
                
                # Save to cache for future use
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(frame, str(cache_path))
        else:
            # No cache: standard on-the-fly processing
            events = self._read_events(file_path)
            
            # Same filter as the cache-miss path above, so a sample is denoised
            # identically whether or not the cache is in use.
            if self.denoiser is not None:
                events = self.denoiser(events, sensor_size=CIFAR10_DVS_SENSOR_SIZE)
            if self.event_transform is not None and self.tensor_transform is None:
                events = self.event_transform(events)

            if self.representation == "spike_tensor":
                frame = self._events_to_spike_tensor(events)
            else:
                frame = self._events_to_frame(events)

        frame = self.resize(frame)
        if self.canonicalize_orientation:
            frame = torch.rot90(frame, k=1, dims=(-2, -1))
        if self.tensor_transform is not None:
            frame = self.tensor_transform(frame)
        if self.transform is not None:
            frame = self.transform(frame)

        return frame, self.labels[index]
