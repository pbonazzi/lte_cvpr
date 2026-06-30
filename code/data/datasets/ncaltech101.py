import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import struct
from torchvision.transforms import Resize, InterpolationMode


class NCaltech101(Dataset):
    """
    N-Caltech101 dataset loader for binary event data.

    The dataset structure is:
    - Root directory contains 101 category subdirectories
    - Each category contains binary event files (image_XXXX.bin)
    - Each event is 40 bits: [8-bit X, 8-bit Y, 1-bit polarity, 23-bit timestamp]

    Reference:
    Orchard, G.; Cohen, G.; Jayawant, A.; and Thakor, N. "Converting Static Image
    Datasets to Spiking Neuromorphic Datasets Using Saccades", Frontiers in
    Neuromorphic Engineering, 2015
    """

    def __init__(
        self,
        root_path,
        transform=None,
        representation='binary',
        target_size=(32, 32),
        split='train',
        train_split=0.8,
        val_split=0.1,
        seed=42,
        num_time_bins=1,
        event_filter=None,
        event_transform=None,
        tensor_transform=None,
        binning_strategy='duration',
        use_cache=False,
        denoise_filter_time_us=None,
    ):
        """
        Args:
            root_path: Path to the N-Caltech101 root directory
            transform: Optional transform to apply to event frames
            representation: 'histogram' (accumulate counts), 'binary' (presence/absence),
                or 'spike_tensor' (stacked binary event frames across time bins)
            target_size: Target size to resize frames to (height, width)
            split: Which split to use: 'train', 'val', or 'test'
            train_split: Fraction of data to use for training (0.0 to 1.0)
            val_split: Fraction of data to use for validation (0.0 to 1.0)
                Test split will be 1.0 - train_split - val_split
            seed: Random seed for reproducible train/test splits
            event_filter: Optional callable applied to raw events for all splits
            event_transform: Optional callable applied after filtering, typically for train only
            tensor_transform: Optional tensor-level augmentation applied after spike tensor creation
            binning_strategy: How to assign events to temporal bins for spike tensors.
                'duration' uses equal time-width bins across the event stream.
                'event_count' uses equal event-count bins in chronological order.
        """
        self.root_path = Path(root_path)
        self.transform = transform
        self.representation = representation
        self.target_size = target_size
        self.num_time_bins = int(num_time_bins)
        self.event_filter = event_filter
        self.event_transform = event_transform
        self.tensor_transform = tensor_transform
        self.binning_strategy = str(binning_strategy)
        self.use_cache = bool(use_cache)
        self.denoise_filter_time_us = denoise_filter_time_us

        if self.representation not in {'histogram', 'binary', 'spike_tensor'}:
            raise ValueError(f"Unsupported representation: {self.representation}")
        if self.num_time_bins <= 0:
            raise ValueError(f"num_time_bins must be positive, got {self.num_time_bins}")
        if self.binning_strategy not in {'duration', 'event_count'}:
            raise ValueError(f"Unsupported binning_strategy: {self.binning_strategy}")
        if self.use_cache and self.representation != 'spike_tensor':
            raise ValueError("NCaltech101 cache only supports representation='spike_tensor'.")
        if split not in {'train', 'val', 'test'}:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")
        if train_split <= 0 or train_split >= 1:
            raise ValueError(f"train_split must be in (0, 1), got {train_split}")
        if val_split < 0 or val_split >= 1:
            raise ValueError(f"val_split must be in [0, 1), got {val_split}")
        if train_split + val_split >= 1:
            raise ValueError(f"train_split + val_split must be < 1, got {train_split + val_split}")

        if not self.root_path.exists():
            raise ValueError(f"Root path does not exist: {root_path}")

        # N-Caltech101 sensor resolution (240x180)
        self.sensor_height = 180
        self.sensor_width = 240

        # Setup resizing
        self.resize = Resize(target_size, interpolation=InterpolationMode.NEAREST)
        if self.denoise_filter_time_us is not None:
            from data.transforms import Denoise
            self.denoiser = Denoise(filter_time=self.denoise_filter_time_us)
        else:
            self.denoiser = event_filter
        self.cache_dir = self._build_cache_dir() if self.use_cache else None

        # Load all file paths and create category mapping
        self.file_paths = []
        self.labels = []
        self.categories = sorted([d.name for d in self.root_path.iterdir()
                                 if d.is_dir() and not d.name.startswith('.')])
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

        # Collect all files
        all_file_paths = []
        all_labels = []

        for category in self.categories:
            category_path = self.root_path / category
            bin_files = sorted(category_path.glob('*.bin'))

            for bin_file in bin_files:
                all_file_paths.append(bin_file)
                all_labels.append(self.category_to_idx[category])

        # Create a stratified train/val/test split so each class keeps roughly the
        # same ratios across splits.
        rng = np.random.default_rng(seed)
        all_labels_array = np.asarray(all_labels, dtype=np.int64)
        selected_indices = []

        for label_idx in range(len(self.categories)):
            class_indices = np.flatnonzero(all_labels_array == label_idx)
            class_indices = rng.permutation(class_indices)
            
            train_idx = int(len(class_indices) * train_split)
            val_idx = train_idx + int(len(class_indices) * val_split)

            if split == 'train':
                chosen_indices = class_indices[:train_idx]
            elif split == 'val':
                chosen_indices = class_indices[train_idx:val_idx]
            else:  # split == 'test'
                chosen_indices = class_indices[val_idx:]

            selected_indices.extend(chosen_indices.tolist())

        if selected_indices:
            selected_indices = rng.permutation(np.asarray(selected_indices, dtype=np.int64))
        else:
            selected_indices = np.asarray([], dtype=np.int64)

        self.file_paths = [all_file_paths[i] for i in selected_indices]
        self.labels = torch.tensor([all_labels[i] for i in selected_indices], dtype=torch.long)

    def _cache_filter_token(self):
        if self.denoise_filter_time_us is not None:
            filter_value = int(round(float(self.denoise_filter_time_us)))
            return f"denoise-{filter_value}us"
        if self.event_filter is not None:
            return "event-filter"
        return "raw"

    def _build_cache_dir(self):
        cache_root = self.root_path.parent / "NCaltech101-cache"
        cache_name = (
            f"spike_tensor-{self.num_time_bins}bins-"
            f"{self.binning_strategy}-{self._cache_filter_token()}"
        )
        return cache_root / cache_name

    def _get_cache_path(self, file_path):
        if not self.use_cache or self.cache_dir is None:
            return None
        relative_path = Path(file_path).resolve().relative_to(self.root_path.resolve())
        return (self.cache_dir / relative_path).with_suffix(".pt")

    def _build_cached_tensor(self, file_path):
        events = self.read_events_from_bin(file_path)
        if self.denoiser is not None:
            events = self.denoiser(events, sensor_size=(self.sensor_height, self.sensor_width))
        return self.events_to_spike_tensor(events)

    def ensure_cache_entry(self, index, force_regenerate=False):
        if not self.use_cache:
            raise RuntimeError("Cache is not enabled for this dataset instance.")

        cache_path = self._get_cache_path(self.file_paths[index])
        if cache_path is None:
            raise RuntimeError("Cache path is unavailable.")

        if cache_path.exists() and not force_regenerate:
            return "existing", cache_path

        frame = self._build_cached_tensor(self.file_paths[index])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(frame, str(cache_path))
        return "generated", cache_path

    def __len__(self):
        return len(self.labels)

    def read_events_from_bin(self, bin_file):
        """
        Read events from binary file.

        Each event is 40 bits packed as 5 bytes:
        - Byte 0: X address
        - Byte 1: Y address
        - Byte 2: Polarity (bit 7) + Timestamp upper bits
        - Bytes 3-4: Timestamp lower bits

        Returns:
            events: Array with shape [num_events, 4] (x, y, timestamp, polarity)
        """
        with open(bin_file, 'rb') as f:
            raw_data = f.read()

        # Each event is 5 bytes (40 bits)
        n_events = len(raw_data) // 5
        events = np.zeros((n_events, 4), dtype=np.float32)

        for i in range(n_events):
            offset = i * 5
            # Read 5 bytes for this event
            data = struct.unpack('5B', raw_data[offset:offset+5])

            x = data[0]
            y = data[1]
            polarity = (data[2] >> 7) & 0x01  # Bit 7 of byte 2
            timestamp = ((data[2] & 0x7F) << 16) | (data[3] << 8) | data[4]  # 23 bits

            events[i] = [x, y, timestamp, polarity]

        return events

    def events_to_frame(self, events):
        """
        Convert events to a frame representation.

        Args:
            events: Array with shape [num_events, 4] (x, y, timestamp, polarity)

        Returns:
            frame: Tensor with shape [2, H, W]
        """
        if len(events) == 0:
            return torch.zeros((2, self.sensor_height, self.sensor_width), dtype=torch.float32)

        x = events[:, 0].astype(np.int32)
        y = events[:, 1].astype(np.int32)
        pol = events[:, 3].astype(np.int32)

        # Clip coordinates to sensor bounds
        x = np.clip(x, 0, self.sensor_width - 1)
        y = np.clip(y, 0, self.sensor_height - 1)

        # Create separate frames for each polarity
        frame = np.zeros((2, self.sensor_height, self.sensor_width), dtype=np.float32)

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
        """
        Convert an event stream to a stacked temporal tensor with shape
        [2 * num_time_bins, H, W] using binary polarity frames per time bin.
        """
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
            # Collapse repeated events at the same bin/polarity/pixel to a
            # binary presence value.
            spike[bin_idx, 0, y[pos_mask], x[pos_mask]] = 1
            spike[bin_idx, 1, y[neg_mask], x[neg_mask]] = 1

        # LogicTreeNet consumes the B temporal bins as 2B input channels.
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

    def _prepare_events(self, events):
        if self.event_filter is not None:
            events = self.event_filter(events, sensor_size=(self.sensor_height, self.sensor_width))
        if self.event_transform is not None:
            events = self.event_transform(events)
        return np.asarray(events, dtype=np.float32)

    def __getitem__(self, index):
        """
        Returns:
            frame: Event frame with shape [2, H, W] after resizing
            label: Category label (0 to 100)
        """
        file_path = self.file_paths[index]
        if self.use_cache:
            cache_path = self._get_cache_path(file_path)
            if cache_path is not None and cache_path.exists():
                frame = torch.load(cache_path, weights_only=True)
            else:
                frame = self._build_cached_tensor(file_path)
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(frame, str(cache_path))
        else:
            events = self.read_events_from_bin(file_path)
            events = self._prepare_events(events)

            if self.representation == 'spike_tensor':
                frame = self.events_to_spike_tensor(events)
            else:
                frame = self.events_to_frame(events)

        if self.tensor_transform is not None:
            frame = self.tensor_transform(frame)

        frame = self.resize(frame)
        label = self.labels[index]

        if self.transform is not None:
            frame = self.transform(frame)

        return frame, label


def select_top_k_categories(root_path, top_k):
    root_path = Path(root_path)
    if not root_path.exists():
        raise ValueError(f"Root path does not exist: {root_path}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    categories = sorted([d.name for d in root_path.iterdir() if d.is_dir() and not d.name.startswith('.')])
    if top_k > len(categories):
        raise ValueError(f"Requested top_k={top_k}, but dataset only has {len(categories)} classes.")

    counts = np.asarray([len(list((root_path / category).glob('*.bin'))) for category in categories], dtype=np.int64)
    top_labels = np.argsort(counts)[::-1][:top_k].tolist()
    top_class_names = [categories[label] for label in top_labels]
    top_counts = [int(counts[label]) for label in top_labels]
    return top_labels, top_class_names, top_counts
