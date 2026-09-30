"""
Preprocessing script for CIFAR10-DVS dataset.
Generates and caches spike tensor representations to disk.

Usage:
    python data/datasets/cifar10_dvs_preprocess.py \
        --data_path /path/to/data \
        --denoise_filter_time_us 3000.0 \
        --num_time_bins 5 \
        --binning_strategy duration
"""

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.datasets.cifar10_dvs import (
    CIFAR10_DVS_SENSOR_SIZE,
    CIFAR10DVS,
    load_cifar10_dvs_metadata,
)


def create_cache_directory(data_path: Path | str) -> Path:
    """Create cache directory for spike tensor caches."""
    cache_dir = Path(data_path) / "CIFAR10-DVS-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def preprocess_cifar10_dvs(
    data_path: Path | str,
    denoise_filter_time_us: float = 3_000.0,
    num_time_bins: int = 5,
    binning_strategy: str = "duration",
    force_regenerate: bool = False,
):
    """
    Preprocess CIFAR10-DVS dataset and cache spike tensors to disk.
    
    Args:
        data_path: Path to dataset root directory
        denoise_filter_time_us: Denoise filter time in microseconds
        num_time_bins: Number of temporal bins for spike tensor
        binning_strategy: How to assign events to bins ("duration" or "event_count")
        force_regenerate: If True, regenerate all caches even if they exist
    """
    cache_dir = create_cache_directory(data_path)
    
    # Load metadata
    file_paths, labels = load_cifar10_dvs_metadata(data_path)
    
    # Create temporary dataset for on-the-fly preprocessing
    temp_dataset = CIFAR10DVS(
        data_path=data_path,
        indices=np.arange(len(labels), dtype=np.int64),
        representation="spike_tensor",
        target_size=(128, 128),  # Keep original size before resize
        num_time_bins=num_time_bins,
        # Must be passed as denoise_filter_time_us, not as event_filter: the
        # cache filename tag is built from this field, so supplying the denoiser
        # only via event_filter tags the files "denoisenone" and a trainer run
        # with --denoise_filter_time_us never finds them.
        denoise_filter_time_us=denoise_filter_time_us,
        event_transform=None,
        binning_strategy=binning_strategy,
    )
    
    # Preprocess and cache
    total_samples = len(labels)
    cached_count = 0
    regenerated_count = 0
    
    pbar = tqdm(total=total_samples, desc="Preprocessing CIFAR10-DVS", unit="sample")
    
    for idx in range(total_samples):
        file_path = file_paths[idx]
        cache_filename = temp_dataset._get_cache_filename(file_path)
        if cache_filename is None:
            print(f"\nSkipping unknown file path: {file_path}")
            pbar.update(1)
            continue
        cache_path = cache_dir / cache_filename
        
        # Check if cache exists
        if cache_path.exists() and not force_regenerate:
            cached_count += 1
            pbar.update(1)
            continue
        
        # Generate and cache
        try:
            # Mirror CIFAR10DVS.__getitem__'s cache-miss path exactly, denoiser
            # included: generating straight from _read_events skipped it and
            # cached tensors that no trainer setting can reproduce.
            events = temp_dataset._read_events(file_path)
            if temp_dataset.denoiser is not None:
                events = temp_dataset.denoiser(
                    events, sensor_size=CIFAR10_DVS_SENSOR_SIZE
                )
            spike_tensor = temp_dataset._events_to_spike_tensor(events)
            
            torch.save(spike_tensor, str(cache_path))
            regenerated_count += 1
        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            pbar.update(1)
            continue
        
        pbar.update(1)
    
    pbar.close()
    
    print(f"\nPreprocessing complete!")
    print(f"  Total samples: {total_samples}")
    print(f"  Cached (existing): {cached_count}")
    print(f"  Regenerated (new): {regenerated_count}")
    print(f"  Cache directory: {cache_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess CIFAR10-DVS dataset and cache spike tensors"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.getenv("DATA_PATH", "./data"),
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--denoise_filter_time_us",
        type=float,
        default=3_000.0,
        help="Denoise filter time in microseconds",
    )
    parser.add_argument(
        "--num_time_bins",
        type=int,
        default=5,
        help="Number of temporal bins for spike tensor",
    )
    parser.add_argument(
        "--binning_strategy",
        type=str,
        choices=["duration", "event_count"],
        default="duration",
        help="How to assign events to temporal bins",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all caches even if they exist",
    )
    
    args = parser.parse_args()
    
    preprocess_cifar10_dvs(
        data_path=args.data_path,
        denoise_filter_time_us=args.denoise_filter_time_us,
        num_time_bins=args.num_time_bins,
        binning_strategy=args.binning_strategy,
        force_regenerate=args.force,
    )
