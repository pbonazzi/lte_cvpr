import argparse
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets.ncaltech101 import NCaltech101
from data.transforms import (
    Denoise,
    TensorEventTransformCompose,
    TensorRandomFlipLR,
    TensorSpatialJitter,
)
from data.utils import create_output_dirs, seed_worker
from models.logictreenet import LogicTreeNet
from scripts.ncaltech101_topk_common import (
    ClassBalancedBatchSampler,
    FilteredNCaltech101,
    build_class_count_table,
    build_class_weights,
    build_confusion_matrix_heatmap,
    build_prediction_table,
    collect_classifier_predictions,
    collect_prediction_samples,
    evaluate_classifier,
    save_model,
    select_sample_indices,
    select_top_k_labels,
    train_classifier_epoch,
)

PROJECT_NAME = "LGN_Events"
SEED = 15
SENSOR_SIZE = (180, 240)
DEFAULT_NUM_WORKERS = 8


def build_parser(default_binning_strategy="duration"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name for wandb")
    parser.add_argument(
        "--num_time_bins",
        type=int,
        default=9,
        help="Number of temporal bins used to build the spike tensor.",
    )
    parser.add_argument(
        "--weights_init_mode",
        type=str,
        default="residual",
        choices=["residual", "gaussian"],
        help="Gate weight initialization for every logic layer.",
    )
    parser.add_argument(
        "--top_k_classes",
        type=int,
        default=6,
        help="Number of most frequent classes to train on.",
    )
    parser.add_argument(
        "--sample_log_count",
        type=int,
        default=16,
        help="Number of train and val samples to visualize in wandb logs.",
    )
    parser.add_argument(
        "--model_scale",
        type=str,
        default="s",
        choices=["s", "m", "b", "l", "g"],
        help="LogicTreeNet scale.",
    )
    parser.add_argument(
        "--grouping_mode",
        type=str,
        default="balanced",
        choices=("balanced", "legacy_padded"),
        help="Feature grouping mode before GroupSum.",
    )
    parser.add_argument(
        "--binning_strategy",
        type=str,
        default=default_binning_strategy,
        choices=("duration", "event_count"),
        help=(
            "How to assign events to temporal bins. 'duration' uses equal time spans; "
            "'event_count' keeps events time-ordered while balancing the number of events per bin."
        ),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "cosine"],
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=0.0,
        help="Minimum learning rate for cosine scheduler.",
    )
    parser.add_argument("--spatial_jitter_max_shift", type=int, default=1)
    parser.add_argument("--flip_lr_p", type=float, default=0.5)
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
        default=3_000.0,
        help="Filter time passed to tonic.transforms.Denoise.",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Load spike tensors from a precomputed N-Caltech101 cache when available.",
    )
    parser.add_argument(
        "--delete_cache_after_training",
        action="store_true",
        help="Delete the N-Caltech101 cache directory after training and evaluation complete.",
    )
    parser.add_argument("--train_sampler", type=str, default="random", choices=("random", "class_balanced"))
    parser.add_argument(
        "--disable_class_weights",
        action="store_false",
        dest="use_class_weights",
        default=True,
    )
    return parser


def build_train_tensor_transform(config):
    return TensorEventTransformCompose(
        [
            TensorSpatialJitter(max_shift=config.spatial_jitter_max_shift),
            TensorRandomFlipLR(p=config.flip_lr_p),
        ]
    )


def build_run_config(args):
    dataset_name = (
        f"N-Caltech101-top{args.top_k_classes}-fullstream-{args.num_time_bins}bin-spike"
        f"{'-event-count-binning' if args.binning_strategy == 'event_count' else ''}"
        f"{'-no-denoise' if not args.use_denoise else ''}"
    )
    return {
        "epochs": args.epochs,
        "train_size": 0.7,
        "val_size": 0.1,
        "dataset": dataset_name,
        "architecture": "LogicTreeNet",
        "out_classes": args.top_k_classes,
        "model_scale": args.model_scale,
        "seed": SEED,
        "target_size": 64,
        "top_k_classes": args.top_k_classes,
        "sample_log_count": args.sample_log_count,
        "sample_log_interval": 1,
        "sample_log_strategy": "stratified",
        "grouping_mode": args.grouping_mode,
        "weights_init_mode": args.weights_init_mode,
        "representation": "spike_tensor",
        "num_time_bins": args.num_time_bins,
        "binning_strategy": args.binning_strategy,
        "spatial_jitter_max_shift": args.spatial_jitter_max_shift,
        "flip_lr_p": args.flip_lr_p,
        "use_denoise": args.use_denoise,
        "denoise_filter_time_us": args.denoise_filter_time_us,
        "use_cache": args.use_cache,
        "tau_gs": 20,
        "lr_tau_gs": 0,
        "tau_noise": 0,
        "lr_model": 0.02,
        "scheduler": args.scheduler,
        "lr_min": args.lr_min,
        "weight_decay": 0.002,
        "batch_size": 16,
        "num_workers": DEFAULT_NUM_WORKERS,
        "train_sampler": args.train_sampler,
        "use_class_weights": args.use_class_weights,
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
        learn_tau_gs=config.lr_tau_gs == 0,
        input_size=config.target_size,
        grouping_mode=config.grouping_mode,
        weights_init_mode=config.weights_init_mode,
    )


def build_base_datasets(config, data_path, seed):
    if config.use_cache:
        event_filter = None
        denoise_filter_time_us = config.denoise_filter_time_us if config.use_denoise else None
    else:
        event_filter = Denoise(filter_time=config.denoise_filter_time_us) if config.use_denoise else None
        denoise_filter_time_us = None
    dataset_kwargs = dict(
        root_path=os.path.join(data_path, "N-Caltech101/Caltech101"),
        transform=None,
        representation=config.representation,
        target_size=(config.target_size, config.target_size),
        train_split=config.train_size,
        val_split=config.val_size,
        seed=seed,
        num_time_bins=config.num_time_bins,
        binning_strategy=config.binning_strategy,
        event_filter=event_filter,
        use_cache=config.use_cache,
        denoise_filter_time_us=denoise_filter_time_us,
    )
    train_tensor_transform = build_train_tensor_transform(config)
    split_specs = {
        "train": dict(split='train', event_transform=None, tensor_transform=train_tensor_transform),
        "val": dict(split='val', event_transform=None, tensor_transform=None),
        "test": dict(split='test', event_transform=None, tensor_transform=None),
        "train_log": dict(split='train', event_transform=None, tensor_transform=None),
        "val_log": dict(split='val', event_transform=None, tensor_transform=None),
        "test_log": dict(split='test', event_transform=None, tensor_transform=None),
    }
    return {
        split: NCaltech101(**split_kwargs, **dataset_kwargs)
        for split, split_kwargs in split_specs.items()
    }


def resolve_class_subset(config, train_base_dataset):
    selected_labels, class_names, _class_counts = select_top_k_labels(
        train_base_dataset.root_path,
        train_base_dataset.categories,
        config.top_k_classes,
    )
    label_counts = np.bincount(
        train_base_dataset.labels.cpu().numpy(),
        minlength=len(train_base_dataset.categories),
    )
    class_counts = [int(label_counts[label]) for label in selected_labels]
    return selected_labels, class_names, class_counts


def build_filtered_datasets(base_datasets, selected_labels, class_names):
    return {
        split: FilteredNCaltech101(base_dataset, selected_labels, class_names)
        for split, base_dataset in base_datasets.items()
    }


def build_loader_kwargs(config, generator):
    loader_kwargs = dict(
        batch_size=config.batch_size,
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


def build_train_loader(dataset, config, generator):
    loader_kwargs = build_loader_kwargs(config, generator)
    if config.train_sampler != "class_balanced":
        return DataLoader(dataset, shuffle=True, **loader_kwargs)

    batch_size = loader_kwargs.pop("batch_size")
    loader_kwargs.pop("drop_last", None)
    return DataLoader(
        dataset,
        batch_sampler=ClassBalancedBatchSampler(
            dataset.labels,
            batch_size=batch_size,
            drop_last=True,
            seed=int(config.seed),
        ),
        **loader_kwargs,
    )


def build_optimizer(model, config):
    if config.lr_tau_gs == 0:
        other_params = [p for name, p in model.named_parameters() if name != "group_sum.tau"]
        params = [
            {"params": other_params, "lr": config.lr_model},
            {"params": [model.group_sum.tau], "lr": config.lr_tau_gs},
        ]
    else:
        params = [{"params": model.parameters(), "lr": config.lr_model}]
    return torch.optim.AdamW(params, weight_decay=config.weight_decay)


def build_scheduler(optimizer, config):
    if config.scheduler != "cosine":
        return None

    total_epochs = max(int(config.epochs), 1)
    lr_lambdas = []
    for param_group in optimizer.param_groups:
        base_lr = float(param_group["lr"])
        if base_lr <= 0.0:
            lr_lambdas.append(lambda epoch: 1.0)
            continue

        min_lr = min(float(config.lr_min), base_lr)
        min_factor = min_lr / base_lr

        def cosine_lambda(epoch, min_factor=min_factor, total_epochs=total_epochs):
            progress = min(max(epoch, 0), total_epochs) / total_epochs
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        lr_lambdas.append(cosine_lambda)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)


def build_lr_log_payload(optimizer):
    lr_payload = {"lr": float(optimizer.param_groups[0]["lr"])}
    if len(optimizer.param_groups) > 1:
        lr_payload["lr_group_sum_tau"] = float(optimizer.param_groups[1]["lr"])
    return lr_payload


def build_selected_class_table(class_names, class_counts):
    table = wandb.Table(columns=["class_idx", "class_name", "train_count"])
    for class_idx, (class_name, class_count) in enumerate(zip(class_names, class_counts)):
        table.add_data(class_idx, class_name, class_count)
    return table


def maybe_add_sample_tables(
    log_payload,
    epoch,
    config,
    model,
    datasets,
    sample_indices,
    class_names,
    device,
):
    should_log_samples = (
        config.sample_log_count > 0
        and ((epoch + 1) % config.sample_log_interval == 0 or epoch == 0 or epoch + 1 == config.epochs)
    )
    if not should_log_samples:
        return

    log_specs = {
        "train": dict(dataset=datasets["train_log"]),
        "val": dict(dataset=datasets["val_log"]),
    }
    for split, spec in log_specs.items():
        records = collect_prediction_samples(
            model,
            spec["dataset"],
            sample_indices[split],
            class_names,
            device,
            split,
        )
        log_payload[f"{split}_prediction_samples"] = build_prediction_table(records)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    config,
    datasets,
    class_names,
    device,
):
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_epoch = 0
    best_accuracy = 0.0
    best_state_dict = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    sample_indices = {
        "train": select_sample_indices(
            datasets["train_log"].labels,
            config.sample_log_count,
            config.sample_log_strategy,
            seed=SEED,
        ),
        "val": select_sample_indices(
            datasets["val_log"].labels,
            config.sample_log_count,
            config.sample_log_strategy,
            seed=SEED + 1,
        ),
    }

    progress_bar = tqdm(range(config.epochs), desc="training epochs")
    for epoch in progress_bar:
        epoch_train_loss, epoch_train_acc = train_classifier_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        epoch_val_loss, epoch_val_acc = evaluate_classifier(
            model,
            val_loader,
            criterion,
            device,
        )

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)

        if epoch_val_acc > best_accuracy:
            best_accuracy = epoch_val_acc
            best_epoch = epoch
            best_state_dict = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

        log_payload = {
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "train_accuracy": epoch_train_acc,
            "val_accuracy": epoch_val_acc,
            "tau_gs": model.group_sum.tau.detach().item(),
        }
        log_payload.update(build_lr_log_payload(optimizer))
        maybe_add_sample_tables(
            log_payload,
            epoch,
            config,
            model,
            datasets,
            sample_indices,
            class_names,
            device,
        )
        wandb.log(log_payload)
        if scheduler is not None:
            scheduler.step()

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

    return best_state_dict, best_epoch, history


def log_dataset_summary(datasets, class_names, class_counts):
    wandb.log(
        {
            "selected_top_classes": build_selected_class_table(class_names, class_counts),
            "train_class_distribution": build_class_count_table(datasets["train_log"].labels, class_names),
            "val_class_distribution": build_class_count_table(datasets["val_log"].labels, class_names),
        }
    )


def log_final_metrics(model, test_loader, criterion, config, history, best_epoch, class_names, device):
    test_loss, test_acc = evaluate_classifier(
        model,
        test_loader,
        criterion,
        device,
    )
    test_y_true, test_y_pred = collect_classifier_predictions(
        model,
        test_loader,
        device,
    )
    wandb.log(
        {
            "best_epoch": best_epoch + 1,
            "best_val_loss": history["val_loss"][best_epoch],
            "best_val_accuracy": history["val_accuracy"][best_epoch],
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_confusion_matrix_counts": build_confusion_matrix_heatmap(
                test_y_true,
                test_y_pred,
                class_names,
                title="Final Test Confusion Matrix",
                normalize=False,
            ),
            "test_confusion_matrix_normalized": build_confusion_matrix_heatmap(
                test_y_true,
                test_y_pred,
                class_names,
                title="Final Test Confusion Matrix (Normalized)",
                normalize=True,
            ),
        }
    )


def main(default_binning_strategy="duration"):
    args = build_parser(default_binning_strategy=default_binning_strategy).parse_args()

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
        base_datasets = build_base_datasets(config, data_path, SEED)
        selected_labels, class_names, class_counts = resolve_class_subset(config, base_datasets["train"])

        print(f"selected top-{config.top_k_classes} classes = {list(zip(class_names, class_counts))}")
        class_weights = build_class_weights(class_counts, device)
        print(f"cross entropy class weights = {class_weights.detach().cpu().tolist()}")
        print(f"train sampler = {config.train_sampler}")
        print(f"use class weights = {bool(config.use_class_weights)}")

        datasets = build_filtered_datasets(base_datasets, selected_labels, class_names)
        loader_kwargs = build_loader_kwargs(config, generator)
        train_loader = build_train_loader(datasets["train"], config, generator)
        val_loader = DataLoader(datasets["val"], shuffle=False, **loader_kwargs)
        test_loader = DataLoader(datasets["test"], shuffle=False, **loader_kwargs)
        criterion = nn.CrossEntropyLoss(weight=class_weights if config.use_class_weights else None)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)

        wandb.watch(model, criterion, log="all", log_freq=1000)
        log_dataset_summary(datasets, class_names, class_counts)

        best_state_dict, best_epoch, history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            config,
            datasets,
            class_names,
            device,
        )

        print("training finished")
        model.load_state_dict(best_state_dict)
        print(f"restored best weights from epoch {best_epoch + 1}")

        log_final_metrics(
            model,
            test_loader,
            criterion,
            config,
            history,
            best_epoch,
            class_names,
            device,
        )

        label, path_saved = save_model(model, ckpt_path)
        print(f"model saved to {path_saved}")

        model_artifact = wandb.Artifact(label, type="model", metadata=dict(config))
        model_artifact.add_file(path_saved)
        wandb.save(path_saved, base_path=ckpt_path)
        wandb.log_artifact(model_artifact)

        if args.delete_cache_after_training and config.use_cache:
            cache_dir = base_datasets["train"].cache_dir
            if cache_dir is not None and cache_dir.exists():
                shutil.rmtree(cache_dir)
                print(f"cache directory deleted: {cache_dir}")

        print("wandb log completed\n------------------------")


if __name__ == "__main__":
    main()
