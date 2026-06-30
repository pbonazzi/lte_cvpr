import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize


NMNIST_SENSOR_HEIGHT = 34
NMNIST_SENSOR_WIDTH = 34


class NMNIST(Dataset):
    """
    N-MNIST dataset loader for binary event files.

    Expected dataset structure:
    - root_path/
        - Train/
            - 0/ ... 9/
        - Test/
            - 0/ ... 9/

    Each sample is stored as a `.bin` file where every event occupies 5 bytes:
    [8-bit x, 8-bit y, 1-bit polarity + 23-bit timestamp].
    """

    def __init__(
        self,
        root_path,
        transform=None,
        representation="binary",
        target_size=(34, 34),
        split="train",
        val_split=0.1,
        seed=42,
        num_time_bins=1,
        event_filter=None,
        event_transform=None,
        tensor_transform=None,
        binning_strategy="duration",
        use_cache=False,
        denoise_filter_time_us=None,
    ):
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

        if self.representation not in {
            "histogram",
            "binary",
            "spike_tensor",
        }:
            raise ValueError(f"Unsupported representation: {self.representation}")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")
        if val_split < 0 or val_split >= 1:
            raise ValueError(f"val_split must be in [0, 1), got {val_split}")
        if self.num_time_bins <= 0:
            raise ValueError(f"num_time_bins must be positive, got {self.num_time_bins}")
        if self.binning_strategy not in {"duration", "event_count"}:
            raise ValueError(f"Unsupported binning_strategy: {self.binning_strategy}")
        if self.use_cache and self.representation != "spike_tensor":
            raise ValueError("NMNIST cache only supports representation='spike_tensor'.")
        if not self.root_path.exists():
            raise ValueError(f"Root path does not exist: {root_path}")

        self.sensor_height = NMNIST_SENSOR_HEIGHT
        self.sensor_width = NMNIST_SENSOR_WIDTH
        self.resize = Resize(target_size, interpolation=InterpolationMode.NEAREST)

        if self.denoise_filter_time_us is not None:
            from data.transforms import Denoise

            self.denoiser = Denoise(filter_time=self.denoise_filter_time_us)
        else:
            self.denoiser = event_filter

        self.cache_dir = self._build_cache_dir() if self.use_cache else None

        split_dir_name = "Test" if split == "test" else "Train"
        self.split_root = self.root_path / split_dir_name
        if not self.split_root.exists():
            raise ValueError(f"N-MNIST split root does not exist: {self.split_root}")

        self.categories = sorted(
            [d.name for d in self.split_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        )
        self.category_to_idx = {category: index for index, category in enumerate(self.categories)}

        all_file_paths = []
        all_labels = []
        for category in self.categories:
            class_dir = self.split_root / category
            for bin_file in sorted(class_dir.glob("*.bin")):
                all_file_paths.append(bin_file)
                all_labels.append(self.category_to_idx[category])

        if split == "test":
            selected_indices = np.arange(len(all_file_paths), dtype=np.int64)
        else:
            rng = np.random.default_rng(seed)
            all_labels_array = np.asarray(all_labels, dtype=np.int64)
            selected_indices = []

            for label_idx in range(len(self.categories)):
                class_indices = np.flatnonzero(all_labels_array == label_idx)
                class_indices = rng.permutation(class_indices)
                val_count = int(len(class_indices) * val_split)

                if split == "val":
                    chosen_indices = class_indices[:val_count]
                else:
                    chosen_indices = class_indices[val_count:]

                selected_indices.extend(chosen_indices.tolist())

            if selected_indices:
                selected_indices = rng.permutation(np.asarray(selected_indices, dtype=np.int64))
            else:
                selected_indices = np.asarray([], dtype=np.int64)

        self.file_paths = [all_file_paths[index] for index in selected_indices]
        self.labels = torch.tensor([all_labels[index] for index in selected_indices], dtype=torch.long)

    def _cache_filter_token(self):
        if self.denoise_filter_time_us is not None:
            filter_value = int(round(float(self.denoise_filter_time_us)))
            return f"denoise-{filter_value}us"
        if self.event_filter is not None:
            return "event-filter"
        return "raw"

    def _build_cache_dir(self):
        cache_root = self.root_path.parent / "N-MNIST-cache"
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
        raw = np.frombuffer(Path(bin_file).read_bytes(), dtype=np.uint8)
        if raw.size % 5 != 0:
            raise ValueError(f"Invalid N-MNIST file {bin_file}: byte count {raw.size} is not divisible by 5.")

        raw = raw.reshape(-1, 5)
        events = np.empty((raw.shape[0], 4), dtype=np.float32)
        events[:, 0] = raw[:, 0]
        events[:, 1] = raw[:, 1]
        events[:, 3] = (raw[:, 2] >> 7) & 0x01
        events[:, 2] = (
            ((raw[:, 2] & 0x7F).astype(np.uint32) << 16)
            | (raw[:, 3].astype(np.uint32) << 8)
            | raw[:, 4].astype(np.uint32)
        ).astype(np.float32)
        return events

    def events_to_frame(self, events):
        if len(events) == 0:
            return torch.zeros((2, self.sensor_height, self.sensor_width), dtype=torch.float32)

        x = np.clip(events[:, 0].astype(np.int32), 0, self.sensor_width - 1)
        y = np.clip(events[:, 1].astype(np.int32), 0, self.sensor_height - 1)
        pol = events[:, 3].astype(np.int32)

        frame = np.zeros((2, self.sensor_height, self.sensor_width), dtype=np.float32)
        pos_mask = pol == 1
        neg_mask = pol == 0

        if self.representation == "histogram":
            np.add.at(frame[0], (y[pos_mask], x[pos_mask]), 1)
            np.add.at(frame[1], (y[neg_mask], x[neg_mask]), 1)
        elif self.representation == "binary":
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
            # Repeated events at the same location are collapsed to binary
            # presence.
            spike[bin_idx, 0, y[pos_mask], x[pos_mask]] = 1
            spike[bin_idx, 1, y[neg_mask], x[neg_mask]] = 1

        # LogicTreeNet receives the temporal bins as stacked 2B channels.
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

    def _prepare_events(self, events):
        if self.event_filter is not None:
            events = self.event_filter(events, sensor_size=(self.sensor_height, self.sensor_width))
        if self.event_transform is not None:
            events = self.event_transform(events)
        return np.asarray(events, dtype=np.float32)

    def __getitem__(self, index):
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

            if self.representation == "spike_tensor":
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
