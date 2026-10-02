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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets.nmnist import NMNIST
from data.transforms import DATASET_TRANSFORM, Denoise
from data.utils import create_output_dirs, seed_worker
from models.logictreenet import LogicTreeNet

PROJECT_NAME = "Nmnist"
SEED = 15
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
    parser.add_argument("--epochs", type=int, default=100)
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
        help="Minimum learning rate for cosine scheduler; defaults to 1%% of lr_model when omitted.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=32,
        help="Square input size after resizing the N-MNIST frames.",
    )
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio carved from Train.")
    parser.add_argument(
        "--no_denoise",
        action="store_false",
        dest="use_denoise",
        default=True,
        help="Disable tonic.transforms.Denoise and feed raw events directly into the spike-tensor pipeline.",
    )
    parser.add_argument(
        "--denoise_filter_time_us",
        type=float,
        default=3_000.0,
        help="Filter time passed to tonic.transforms.Denoise before train/val/test processing.",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Load spike tensors from a precomputed N-MNIST cache when available.",
    )
    parser.add_argument(
        "--delete_cache_after_training",
        action="store_true",
        help="Delete the N-MNIST cache directory after training and evaluation complete.",
    )
    parser.add_argument("--tau_gs", type=float, default=20.0,
                        help="GroupSum temperature: each class score is divided by it.")
    parser.add_argument("--tau_noise", type=float, default=0.0,
                        help="Gumbel noise scale for sampling hard gates during training (0 = off: softmax gates).")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing of the cross-entropy loss (0 = off).")
    parser.add_argument("--lr_model", type=float, default=0.02,
                        help="Learning rate of the gate weights.")
    parser.add_argument("--seed", type=int, default=15,
                        help="Seed for initialisation, data order and augmentation; the data split stays fixed.")
    return parser


def build_run_config(args):
    dataset_name = (
        f"N-MNIST-spike_tensor-{args.num_time_bins}bin"
        f"{'-event-count-binning' if args.binning_strategy == 'event_count' else ''}"
        f"{'-no-denoise' if not args.use_denoise else ''}"
    )
    return {
        "epochs": args.epochs,
        "dataset": dataset_name,
        "architecture": "LogicTreeNet",
        "representation": "spike_tensor",
        "out_classes": 10,
        "model_scale": args.model_scale,
        "seed": args.seed,
        "input_size": args.input_size,
        "num_time_bins": args.num_time_bins,
        "binning_strategy": args.binning_strategy,
        "val_split": args.val_split,
        "use_denoise": args.use_denoise,
        "denoise_filter_time_us": args.denoise_filter_time_us,
        "use_cache": args.use_cache,
        "tau_gs": args.tau_gs,
        "lr_tau_gs": 0,
        "tau_noise": args.tau_noise,
        "label_smoothing": args.label_smoothing,
        "lr_model": args.lr_model,
        "scheduler": args.scheduler,
        "lr_min": 0.0002 if args.lr_min is None else args.lr_min,
        "weight_decay": 0.002,
        "train_batch_size": 64,
        "eval_batch_size": 128,
        "num_workers": DEFAULT_NUM_WORKERS,
        "warmup_epochs": 10,
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
    dataset_root = Path(data_path) / "N-MNIST"
    if config.use_cache:
        event_filter = None
        denoise_filter_time_us = config.denoise_filter_time_us if config.use_denoise else None
    else:
        event_filter = Denoise(filter_time=config.denoise_filter_time_us) if config.use_denoise else None
        denoise_filter_time_us = None

    dataset_kwargs = dict(
        root_path=dataset_root,
        representation="spike_tensor",
        target_size=(config.input_size, config.input_size),
        val_split=config.val_split,
        seed=SEED,  # validation split, fixed across --seed
        num_time_bins=config.num_time_bins,
        binning_strategy=config.binning_strategy,
        event_filter=event_filter,
        use_cache=config.use_cache,
        denoise_filter_time_us=denoise_filter_time_us,
    )

    return {
        "train": NMNIST(
            split="train",
            transform=DATASET_TRANSFORM["N-MNIST"]["train"],
            **dataset_kwargs,
        ),
        "val": NMNIST(
            split="val",
            transform=DATASET_TRANSFORM["N-MNIST"]["val"],
            **dataset_kwargs,
        ),
        "test": NMNIST(
            split="test",
            transform=DATASET_TRANSFORM["N-MNIST"]["val"],
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
            T_max=max(int(config.epochs) - int(config.warmup_epochs), 1),
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{model.__class__.__name__}_{timestamp}.pth"
    location = os.path.join(ckpt_path, label)
    torch.save(model.state_dict(), location)
    return label[:-4], location


def maybe_delete_cache(datasets):
    cache_dirs = {dataset.cache_dir for dataset in datasets.values() if getattr(dataset, "cache_dir", None) is not None}
    for cache_dir in cache_dirs:
        if cache_dir is not None and cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"Deleted cache directory: {cache_dir}")


def main():
    args = build_parser().parse_args()

    load_dotenv()

    output_path = Path(os.getenv("OUTPUT_PATH", "./outputs"))
    data_path = Path(os.getenv("DATA_PATH", "./data"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = seed_everything(args.seed)
    run_config = build_run_config(args)

    print(f"device = {device}")
    print(f"data path = {data_path}")

    wandb.login()
    with wandb.init(project=os.getenv("WANDB_PROJECT", PROJECT_NAME), config=run_config, mode=args.wandb_mode, name=args.run_name) as run:
        config = wandb.config

        ckpt_path, config_path, _log_path = create_output_dirs(os.path.join(output_path, run.name))
        json.dump(dict(config), open(os.path.join(config_path, "config.json"), "w"), indent=4)

        datasets = build_datasets(config, data_path)
        train_loader = DataLoader(
            datasets["train"],
            shuffle=True,
            **build_loader_kwargs(config, generator, batch_size=config.train_batch_size),
        )
        val_loader = DataLoader(
            datasets["val"],
            shuffle=False,
            **build_loader_kwargs(config, generator, batch_size=config.eval_batch_size),
        )
        test_loader = DataLoader(
            datasets["test"],
            shuffle=False,
            **build_loader_kwargs(config, generator, batch_size=config.eval_batch_size),
        )

        print(
            f"dataset sizes: train={len(datasets['train'])}, val={len(datasets['val'])}, test={len(datasets['test'])}"
        )

        model = build_model(config).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)

        wandb.watch(model, criterion, log="all", log_freq=1000)

        best_val_accuracy = 0.0
        best_epoch = 0
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        progress = tqdm(range(config.epochs), desc="training epochs")
        for epoch in progress:
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "tau_gs": scalar_value(model.group_sum.tau),
                    "lr": current_lr,
                }
            )

            progress.set_postfix(
                train_loss=f"{train_loss:.3f}",
                train_acc=f"{train_acc:.3f}",
                val_loss=f"{val_loss:.3f}",
                val_acc=f"{val_acc:.3f}",
                lr=f"{current_lr:.5f}",
            )

            if scheduler is not None and epoch >= config.warmup_epochs:
                scheduler.step()

        print("training finished")

        model.load_state_dict(best_state)
        print(f"restored best weights from epoch {best_epoch + 1}")

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        wandb.log(
            {
                "best_epoch": best_epoch + 1,
                "best_val_accuracy": best_val_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
            }
        )

        _label, model_path = save_model(model, ckpt_path)
        print(f"model saved to {model_path}")

        if args.delete_cache_after_training:
            maybe_delete_cache(datasets)


if __name__ == "__main__":
    main()
