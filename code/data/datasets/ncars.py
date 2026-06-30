import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path

NCARS_SENSOR_HEIGHT = 100
NCARS_SENSOR_WIDTH = 120

class NCars(Dataset):
    """
    N-Cars dataset loader for parsed event data with lazy loading.
    Converts events to frame representation compatible with image transforms.

    The dataset structure is:
    - train/val/test splits in separate directories
    - Each sample is a sequence folder containing:
        - events.txt: Event data with columns [x, y, timestamp, polarity]
        - is_car.txt: Binary label (1 for car, 0 for not car)
    """

    def __init__(
        self,
        split_path,
        transform=None,
        representation='binary',
        num_time_bins=1,
        event_filter=None,
        event_transform=None,
        event_frame_filter=None,
        frame_filter=None,
        binning_strategy='duration',
        target_size=64,
        use_cache=False,
        denoise_filter_time_us=None,
    ):
        """
        Args:
            split_path: Path to the split directory (train/test/val)
            transform: Optional transform to apply to event frames
            representation: 'histogram' (accumulate counts) or 'binary' (presence/absence)
            target_size: Target resolution for frames (default: 64)
        """
        self.split_path = Path(split_path)
        self.transform = transform
        self.event_filter = event_filter
        self.event_transform = event_transform
        self.event_frame_filter = event_frame_filter
        self.frame_filter = frame_filter

        # N-Cars event coordinates are defined on a 120x100 sensor.
        self.sensor_height = NCARS_SENSOR_HEIGHT
        self.sensor_width = NCARS_SENSOR_WIDTH
        self.target_size = int(target_size)
        self.representation = representation
        self.num_time_bins = int(num_time_bins)
        self.binning_strategy = str(binning_strategy)
        self.split_name = Path(split_path).name

        if self.representation not in {'histogram', 'binary', 'spike_tensor'}:
            raise ValueError(f"Unsupported representation: {self.representation}")
        if self.num_time_bins <= 0:
            raise ValueError(f"num_time_bins must be positive, got {self.num_time_bins}")
        if self.binning_strategy not in {'duration', 'event_count'}:
            raise ValueError(f"Unsupported binning_strategy: {self.binning_strategy}")
        if use_cache and self.representation != 'spike_tensor':
            raise ValueError("NCars cache only supports representation='spike_tensor'.")
        if use_cache and self.event_frame_filter is not None:
            raise ValueError("NCars cache is incompatible with event_frame_filter because it stores spike tensors.")

        if not self.split_path.exists():
            raise ValueError(f"Split path does not exist: {split_path}")

        self.use_cache = bool(use_cache)
        self.denoise_filter_time_us = denoise_filter_time_us
        self.cache_dir = self._build_cache_dir() if self.use_cache else None
        self.tensor_transform = self._build_tensor_transform(event_transform) if self.use_cache else None
        if self.denoise_filter_time_us is not None:
            from data.transforms import Denoise
            self.denoiser = Denoise(filter_time=self.denoise_filter_time_us)
        else:
            self.denoiser = event_filter

        # Get all sequence directories and sort by sequence number
        sequence_dirs = [d for d in self.split_path.iterdir() if d.is_dir()]
        sequence_dirs = sorted(sequence_dirs, key=lambda x: int(x.name.split('_')[1]))

        # Store paths and load labels only (labels are small)
        self.sequence_paths = []
        self.labels = []

        for seq_dir in sequence_dirs:
            events_file = seq_dir / "events.txt"
            label_file = seq_dir / "is_car.txt"

            if not events_file.exists() or not label_file.exists():
                continue

            self.sequence_paths.append(events_file)

            # Load label (single integer, fast)
            with open(label_file, 'r') as f:
                label = int(f.read().strip())
            self.labels.append(label)

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def _build_tensor_transform(self, event_transform):
        if event_transform is None:
            return None

        from data.transforms import (
            EventTransformCompose,
            RandomFlipLR,
            SpatialJitter,
            TensorEventTransformCompose,
            TensorRandomFlipLR,
            TensorSpatialJitter,
        )

        if not isinstance(event_transform, EventTransformCompose):
            return None

        tensor_transforms = []
        for transform in event_transform.transforms:
            if isinstance(transform, SpatialJitter):
                tensor_transforms.append(TensorSpatialJitter(max_shift=transform.max_shift))
            elif isinstance(transform, RandomFlipLR):
                tensor_transforms.append(TensorRandomFlipLR(p=transform.p))

        if not tensor_transforms:
            return None

        return TensorEventTransformCompose(tensor_transforms)

    def _cache_filter_token(self):
        if self.denoise_filter_time_us is not None:
            filter_value = int(round(float(self.denoise_filter_time_us)))
            return f"denoise-{filter_value}us"
        if self.event_filter is not None:
            return "event-filter"
        return "raw"

    def _build_cache_dir(self):
        dataset_root = self.split_path.parent
        cache_root = dataset_root / "NCars-cache"
        cache_name = (
            f"spike_tensor-{self.num_time_bins}bins-"
            f"{self.binning_strategy}-{self._cache_filter_token()}"
        )
        return cache_root / cache_name

    def _get_cache_path(self, sequence_path):
        if not self.use_cache or self.cache_dir is None:
            return None
        sequence_name = Path(sequence_path).parent.name
        return self.cache_dir / self.split_name / f"{sequence_name}.pt"

    @staticmethod
    def _read_events(sequence_path):
        return np.fromfile(
            sequence_path,
            dtype=np.float32,
            sep=' '
        ).reshape(-1, 4)

    def _build_cached_tensor(self, sequence_path):
        events = self._read_events(sequence_path)
        if self.denoiser is not None:
            events = self.denoiser(events, sensor_size=(self.sensor_height, self.sensor_width))
        return self.events_to_spike_tensor(events)

    def ensure_cache_entry(self, index, force_regenerate=False):
        if not self.use_cache:
            raise RuntimeError("Cache is not enabled for this dataset instance.")

        cache_path = self._get_cache_path(self.sequence_paths[index])
        if cache_path is None:
            raise RuntimeError("Cache path is unavailable.")

        if cache_path.exists() and not force_regenerate:
            return "existing", cache_path

        frame = self._build_cached_tensor(self.sequence_paths[index])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(frame, str(cache_path))
        return "generated", cache_path

    def events_to_frame(self, events):
        """
        Convert events to a frame representation.

        Args:
            events: Array with shape [num_events, 4] (x, y, t, polarity)

        Returns:
            frame: Tensor with shape [3, H, W] compatible with image transforms
        """
        if len(events) == 0:
            return torch.zeros((2, self.sensor_height, self.sensor_width), dtype=torch.float32)

        # Use coordinates directly (already in camera frame)
        x = events[:, 0].astype(np.int32)
        y = events[:, 1].astype(np.int32)
        pol = events[:, 3]

        # N-Cars has a fixed sensor resolution; keep frames aligned even if
        # filtering removes border events or all events in a sample.
        frame_width = self.sensor_width
        frame_height = self.sensor_height
        x = np.clip(x, 0, frame_width - 1)
        y = np.clip(y, 0, frame_height - 1)

        # Create separate frames for each polarity
        frame = np.zeros((2, frame_height, frame_width), dtype=np.float32) 

        # Accumulate events using np.add.at
        pos_mask = pol == 1
        neg_mask = pol == 0
        if self.representation == 'histogram':
            np.add.at(frame[0], (y[pos_mask], x[pos_mask]), 1)
            np.add.at(frame[1], (y[neg_mask], x[neg_mask]), 1)
        elif self.representation == 'binary':
            frame[0][y[pos_mask], x[pos_mask]] = 1
            frame[1][y[neg_mask], x[neg_mask]] = 1  
        return torch.from_numpy(frame)

    def events_to_spike_tensor(self, events):
        # Build one binary occupancy map per time bin and polarity.
        spike = np.zeros(
            (self.num_time_bins, 2, self.sensor_height, self.sensor_width),
            dtype=np.float32,
        )

        if len(events) == 0:
            return torch.from_numpy(spike.reshape(2 * self.num_time_bins, self.sensor_height, self.sensor_width))

        events = np.asarray(events, dtype=np.float32)
        events = self._sort_events_by_time(events)
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
            # Store polarity presence, not event counts.
            spike[bin_idx, 0, y[pos_mask], x[pos_mask]] = 1
            spike[bin_idx, 1, y[neg_mask], x[neg_mask]] = 1

        # Merge temporal and polarity axes into the channel dimension.
        return torch.from_numpy(spike.reshape(2 * self.num_time_bins, self.sensor_height, self.sensor_width))

    @staticmethod
    def _sort_events_by_time(events):
        if len(events) > 1 and np.any(events[1:, 2] < events[:-1, 2]):
            order = np.argsort(events[:, 2], kind="mergesort")
            events = events[order]
        return events

    def _compute_bin_ids(self, events):
        if len(events) == 0:
            return np.zeros(0, dtype=np.int64)

        if self.binning_strategy == 'event_count':
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

    def _resize_with_aspect_and_pad(self, frame):
        if frame.ndim != 3:
            raise ValueError(f"Expected frame with shape [C, H, W], got {tuple(frame.shape)}")

        _, height, width = frame.shape
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid frame shape {tuple(frame.shape)}")

        scale = float(self.target_size) / float(max(height, width))
        resized_height = max(1, int(round(height * scale)))
        resized_width = max(1, int(round(width * scale)))

        resized = F.interpolate(
            frame.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="nearest",
        ).squeeze(0)

        pad_height = self.target_size - resized_height
        pad_width = self.target_size - resized_width
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom))

    def __getitem__(self, index):
        """
        Returns:
            frame: Event frame with shape [3, H, W]
            label: Binary label (0 or 1)
        """
        sequence_path = self.sequence_paths[index]

        if self.use_cache:
            cache_path = self._get_cache_path(sequence_path)
            if cache_path is not None and cache_path.exists():
                frame = torch.load(cache_path, weights_only=True)
            else:
                frame = self._build_cached_tensor(sequence_path)
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(frame, str(cache_path))
        else:
            events = self._read_events(sequence_path)

            if self.event_frame_filter is not None:
                frame = self.event_frame_filter(events, sensor_size=(self.sensor_height, self.sensor_width))
            else:
                if self.event_filter is not None:
                    events = self.event_filter(events, sensor_size=(self.sensor_height, self.sensor_width))
                if self.event_transform is not None:
                    events = self.event_transform(events)

                if self.representation == 'spike_tensor':
                    frame = self.events_to_spike_tensor(events)
                else:
                    frame = self.events_to_frame(events)

        if self.frame_filter is not None:
            frame = self.frame_filter(frame)
        if self.use_cache and self.tensor_transform is not None:
            frame = self.tensor_transform(frame)

        frame = self._resize_with_aspect_and_pad(frame)
        label = self.labels[index]

        if self.transform is not None:
            frame = self.transform(frame)

        return frame, label
