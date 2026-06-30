"""
Preprocessing script for the N-MNIST dataset.
Generates and caches spike tensor representations to disk.

Usage:
    python data/datasets/nmnist_preprocess.py \
        --data_path /path/to/data \
        --denoise_filter_time_us 3000.0 \
        --num_time_bins 9 \
        --binning_strategy duration
"""

import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.datasets.nmnist import NMNIST


DEFAULT_SPLITS = ("train", "val", "test")


def preprocess_nmnist(
    data_path: Path | str,
    denoise_filter_time_us: float = 3_000.0,
    num_time_bins: int = 9,
    binning_strategy: str = "duration",
    use_denoise: bool = True,
    val_split: float = 0.1,
    seed: int = 15,
    splits=DEFAULT_SPLITS,
    force_regenerate: bool = False,
):
    dataset_root = Path(data_path) / "N-MNIST"
    if not dataset_root.exists():
        raise ValueError(f"N-MNIST dataset root does not exist: {dataset_root}")

    split_datasets = {}
    total_samples = 0
    for split in splits:
        dataset = NMNIST(
            root_path=dataset_root,
            representation="spike_tensor",
            target_size=(34, 34),
            split=split,
            val_split=val_split,
            seed=seed,
            num_time_bins=num_time_bins,
            binning_strategy=binning_strategy,
            use_cache=True,
            denoise_filter_time_us=denoise_filter_time_us if use_denoise else None,
        )
        split_datasets[split] = dataset
        total_samples += len(dataset)

    generated_count = 0
    existing_count = 0
    error_count = 0
    last_cache_dir = None

    progress = tqdm(total=total_samples, desc="Preprocessing N-MNIST", unit="sample")
    for split, dataset in split_datasets.items():
        last_cache_dir = dataset.cache_dir
        for index in range(len(dataset)):
            try:
                status, _cache_path = dataset.ensure_cache_entry(index, force_regenerate=force_regenerate)
                if status == "generated":
                    generated_count += 1
                else:
                    existing_count += 1
            except Exception as exc:
                error_count += 1
                print(f"\nError processing split={split} index={index}: {exc}")
            finally:
                progress.update(1)
    progress.close()

    print("\nPreprocessing complete!")
    print(f"  Total samples: {total_samples}")
    print(f"  Cached (existing): {existing_count}")
    print(f"  Regenerated (new): {generated_count}")
    print(f"  Errors: {error_count}")
    if last_cache_dir is not None:
        print(f"  Cache directory: {last_cache_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess N-MNIST and cache denoised spike tensors"
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
        "--no_denoise",
        action="store_false",
        dest="use_denoise",
        default=True,
        help="Disable denoise and cache raw spike tensors instead.",
    )
    parser.add_argument(
        "--num_time_bins",
        type=int,
        default=9,
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
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio carved from the official Train split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=15,
        help="Seed used to build the train/val split",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(DEFAULT_SPLITS),
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to preprocess",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all caches even if they already exist",
    )

    args = parser.parse_args()

    preprocess_nmnist(
        data_path=args.data_path,
        denoise_filter_time_us=args.denoise_filter_time_us,
        num_time_bins=args.num_time_bins,
        binning_strategy=args.binning_strategy,
        use_denoise=args.use_denoise,
        val_split=args.val_split,
        seed=args.seed,
        splits=tuple(args.splits),
        force_regenerate=args.force,
    )
