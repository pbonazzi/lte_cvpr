import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.datasets.ncars import NCars
from data.transforms import (
    ActiveSpatialCrop,
    Denoise,
    EventTransformCompose,
    RandomFlipLR,
    SpatialJitter,
    build_binary_augmentation,
)
from data.utils import create_output_dirs, seed_worker
from models.logictreenet import LogicTreeNet

PROJECT_NAME = "Ncars"
SEED = 15
SENSOR_SIZE = (100, 120)
DEFAULT_NUM_WORKERS = 8


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name for wandb")
    parser.add_argument("--num_time_bins", type=int, default=3, help="Number of temporal bins for the spike tensor.")
    parser.add_argument(
        "--binning_strategy",
        type=str,
        default="duration",
        choices=("duration", "event_count"),
        help="How to assign events to temporal bins.",
    )
    parser.add_argument("--model_scale", type=str, default="s", choices=["s", "m", "b", "l", "g"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=None,
        help="Minimum learning rate for cosine scheduler; defaults to 1% of lr_model when omitted.",
    )
    parser.add_argument("--input_size", type=int, default=32, help="Input size for the model (must be divisible by 16)")
    parser.add_argument("--spatial_jitter_max_shift", type=int, default=1)
    parser.add_argument("--flip_lr_p", type=float, default=0.5)
    parser.add_argument(
        "--active_crop_margin",
        type=int,
        default=None,
        help="Optional margin (in sensor pixels) for cropping to the active event region before resizing.",
    )
    parser.add_argument(
        "--no_denoise",
        action="store_false",
        dest="use_denoise",
        default=True,
        help="Disable tonic.transforms.Denoise and feed raw events directly into the spike-tensor pipeline.",
    )
    parser.add_argument(
        "--denoise_filter_time_us",
        "--denoise_temporal_window_us",
        dest="denoise_filter_time_us",
        type=float,
        default=50_000.0,
        help="Filter time passed to tonic.transforms.Denoise before train/val/test processing.",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Load denoised spike tensors from precomputed NCars cache when available.",
    )
    parser.add_argument(
        "--delete_cache_after_training",
        action="store_true",
        help="Delete the NCars cache directory after training and evaluation complete.",
    )
    parser.add_argument("--tau_gs", type=float, default=20.0,
                        help="GroupSum temperature: each class score is divided by it.")
    parser.add_argument("--affine_degrees", type=float, default=0.0,
                        help="Train-time random rotation range in degrees (0 = off).")
    parser.add_argument("--affine_translate", type=float, default=0.0,
                        help="Train-time random shift as a fraction of width and height (0 = off).")
    parser.add_argument("--affine_scale", type=float, default=0.0,
                        help="Train-time random zoom: scale drawn from [1 - s, 1 + s] (0 = off).")
    parser.add_argument("--erase_p", type=float, default=0.0,
                        help="Probability of erasing a random rectangle of each training sample (0 = off).")
    return parser


def build_train_event_transform(config, sensor_size):
    _height, width = sensor_size
    return EventTransformCompose(
        [
            SpatialJitter(max_shift=config.spatial_jitter_max_shift, sensor_size=sensor_size),
            RandomFlipLR(p=config.flip_lr_p, sensor_width=width, sensor_size=sensor_size),
        ]
    )


def build_run_config(args):
    return {
        "epochs": args.epochs,
        "dataset": f"N_Cars-spike_tensor-{args.num_time_bins}bin"
        f"{'-event-count-binning' if args.binning_strategy == 'event_count' else ''}",
        "architecture": "LogicTreeNet",
        "representation": "spike_tensor",
        "out_classes": 2,
        "model_scale": args.model_scale,
        "seed": SEED,
        "input_size": args.input_size,
        "num_time_bins": args.num_time_bins,
        "binning_strategy": args.binning_strategy,
        "spatial_jitter_max_shift": args.spatial_jitter_max_shift,
        "flip_lr_p": args.flip_lr_p,
        "active_crop_margin": args.active_crop_margin,
        "use_denoise": args.use_denoise,
        "denoise_filter_time_us": args.denoise_filter_time_us,
        "use_cache": args.use_cache,
        "affine_degrees": args.affine_degrees,
        "affine_translate": args.affine_translate,
        "affine_scale": args.affine_scale,
        "erase_p": args.erase_p,
        "tau_gs": args.tau_gs,
        "lr_tau_gs": 0,
        "tau_noise": 0,
        "lr_model": 0.015,
        "scheduler": args.scheduler,
        "lr_min": 0.00015 if args.lr_min is None else args.lr_min,
        "weight_decay": 0.005,
        "train_batch_size": 32,
        "eval_batch_size": 32,
        "num_workers": DEFAULT_NUM_WORKERS,
        "warmup_epochs": 10,  # warmup epochs
    }


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_model(config):
    return LogicTreeNet(
        config.model_scale,
        in_ch=2 * config.num_time_bins,
        out_classes=config.out_classes,
        tau_gs=config.tau_gs,
        tau_noise=config.tau_noise,
        learn_tau_gs=config.lr_tau_gs > 0,
        input_size=config.input_size,
    )


def build_datasets(config, data_path):
    dataset_root = os.path.join(data_path, "N-Cars/N-Cars_parsed")
    if config.use_cache:
        event_filter = None
        denoise_filter_time_us = config.denoise_filter_time_us if config.use_denoise else None
    else:
        event_filter = Denoise(filter_time=config.denoise_filter_time_us) if config.use_denoise else None
        denoise_filter_time_us = None
    frame_filter = (
        ActiveSpatialCrop(margin=config.active_crop_margin)
        if config.active_crop_margin is not None
        else None
    )
    train_event_transform = build_train_event_transform(config, SENSOR_SIZE)
    dataset_kwargs = dict(
        transform=None,
        representation="spike_tensor",
        num_time_bins=config.num_time_bins,
        binning_strategy=config.binning_strategy,
        event_filter=event_filter,
        frame_filter=frame_filter,
        target_size=config.input_size,
        use_cache=config.use_cache,
        denoise_filter_time_us=denoise_filter_time_us,
    )

    return {
        "train": NCars(
            os.path.join(dataset_root, "train"),
            event_transform=train_event_transform,
            **{**dataset_kwargs, "transform": build_binary_augmentation(config.affine_degrees, config.affine_translate, config.affine_scale, config.erase_p)},
        ),
        "val": NCars(
            os.path.join(dataset_root, "val"),
            event_transform=None,
            **dataset_kwargs,
        ),
        "test": NCars(
            os.path.join(dataset_root, "test"),
            event_transform=None,
            **dataset_kwargs,
        ),
    }


def build_loader_kwargs(config, generator, batch_size):
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=config.num_workers,
        generator=generator,
        worker_init_fn=seed_worker,
        drop_last=True,
        pin_memory=True,
    )
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return loader_kwargs


def build_optimizer(model, config):
    if config.lr_tau_gs > 0:
        other_params = [p for name, p in model.named_parameters() if name != "group_sum.tau"]
        params = [
            {"params": other_params, "lr": config.lr_model},
            {"params": [model.group_sum.tau], "lr": config.lr_tau_gs},
        ]
    else:
        params = [{"params": model.parameters(), "lr": config.lr_model}]
    return torch.optim.AdamW(params, weight_decay=config.weight_decay)


def build_scheduler(optimizer, config):
    if config.scheduler == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=config.epochs - config.warmup_epochs,
            eta_min=config.lr_min,
        )
    return None


def scalar_value(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_right = 0
    total_samples = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        batch_loss = criterion(logits, y)
        batch_loss.backward()
        optimizer.step()

        batch_size = y.shape[0]
        total_loss += batch_loss.item() * batch_size
        total_right += (logits.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_right / total_samples


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_right = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            batch_loss = criterion(logits, y)

            batch_size = y.shape[0]
            total_loss += batch_loss.item() * batch_size
            total_right += (logits.argmax(dim=1) == y).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_right / total_samples


def save_model(model, ckpt_path):
    ct = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{model.__class__.__name__}_{ct}.pth"
    location = os.path.join(ckpt_path, label)
    torch.save(model.state_dict(), location)
    return label[:-4], location


def main():
    args = build_parser().parse_args()

    load_dotenv()
    if args.wandb_mode != "disabled":
        wandb.login()

    output_path = Path(os.getenv("OUTPUT_PATH", "./outputs"))
    data_path = Path(os.getenv("DATA_PATH", "./data"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    print(f"data path = {data_path}")

    run_config = build_run_config(args)
    generator = seed_everything(SEED)

    with wandb.init(project=os.getenv("WANDB_PROJECT", PROJECT_NAME), config=run_config, mode=args.wandb_mode, name=args.run_name) as run:
        config = wandb.config

        ckpt_path, config_path, _log_path = create_output_dirs(os.path.join(output_path, run.name))
        with open(os.path.join(config_path, "config.json"), "w") as config_file:
            json.dump(dict(config), config_file, indent=4)

        model = build_model(config).to(device)
        criterion = nn.CrossEntropyLoss()
        datasets = build_datasets(config, data_path)
        train_loader_kwargs = build_loader_kwargs(config, generator, config.train_batch_size)
        eval_loader_kwargs = build_loader_kwargs(config, generator, config.eval_batch_size)
        train_loader = DataLoader(datasets["train"], shuffle=True, **train_loader_kwargs)
        val_loader = DataLoader(datasets["val"], shuffle=False, **eval_loader_kwargs)
        test_loader = DataLoader(datasets["test"], shuffle=False, **eval_loader_kwargs)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)

        wandb.watch(model, criterion, log="all", log_freq=1000)

        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        best_epoch = 0
        best_accuracy = 0.0
        best_state_dict = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

        progress_bar = tqdm(range(config.epochs), desc="training epochs")
        for epoch in progress_bar:
            epoch_train_loss, epoch_train_acc = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
            )
            epoch_val_loss, epoch_val_acc = evaluate(model, val_loader, criterion, device)

            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)
            history["val_accuracy"].append(epoch_val_acc)

            # Learning rate scheduling (skip warmup phase)
            if scheduler is not None and epoch >= config.warmup_epochs:
                scheduler.step()

            if epoch_val_acc > best_accuracy:
                best_accuracy = epoch_val_acc
                best_epoch = epoch
                best_state_dict = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                    "train_accuracy": epoch_train_acc,
                    "val_accuracy": epoch_val_acc,
                    "tau_gs": scalar_value(model.group_sum.tau),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

            previous_train_loss = history["train_loss"][-2] if len(history["train_loss"]) > 1 else None
            progress_bar.set_postfix(
                {
                    "train loss (curr, prev)": (
                        f"{epoch_train_loss:.3f}",
                        f"{previous_train_loss:.3f}" if previous_train_loss is not None else None,
                    ),
                    "val loss": f"{epoch_val_loss:.3f}",
                    "train acc": f"{epoch_train_acc:.3f}",
                    "val acc": f"{epoch_val_acc:.3f}",
                }
            )

        print("training finished")

        model.load_state_dict(best_state_dict)
        print(f"restored best weights from epoch {best_epoch + 1}")

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        wandb.log(
            {
                "best_epoch": best_epoch + 1,
                "best_val_loss": history["val_loss"][best_epoch],
                "best_val_accuracy": history["val_accuracy"][best_epoch],
                "test_loss": test_loss,
                "test_accuracy": test_acc,
            }
        )

        label, path_saved = save_model(model, ckpt_path)
        print(f"model saved to {path_saved}")

        model_artifact = wandb.Artifact(label, type="model", metadata=dict(config))
        model_artifact.add_file(path_saved)
        wandb.save(path_saved, base_path=ckpt_path)
        wandb.log_artifact(model_artifact)
        print("wandb log completed\n------------------------")

        if args.delete_cache_after_training and config.use_cache:
            cache_dir = datasets["train"].cache_dir
            if cache_dir is not None and cache_dir.exists():
                shutil.rmtree(cache_dir)
                print(f"cache directory deleted: {cache_dir}")


if __name__ == "__main__":
    main()
