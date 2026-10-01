import argparse
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path 
import os
from dotenv import load_dotenv

import wandb
import numpy as np 
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader
import torch.nn as nn  

from models.logictreenet import LogicTreeNet
from data.datasets.cifar10_dvs import (
    CIFAR10DVS,
    CIFAR10_DVS_CLASS_NAMES,
    CIFAR10_DVS_SENSOR_SIZE,
    build_cifar10_dvs_splits,
    download_cifar10_dvs,
)
from data.transforms import Denoise, EventTransformCompose, RandomFlipLR, SpatialJitter, build_binary_augmentation
from data.utils import seed_worker, create_output_dirs

CONV_LOGIC_EVAL_BATCH_MULTIPLE = 16
DEFAULT_NUM_WORKERS = 8

def build_train_event_transform(config):
    _height, width = CIFAR10_DVS_SENSOR_SIZE
    return EventTransformCompose(
        [
            SpatialJitter(max_shift=config.spatial_jitter_max_shift, sensor_size=CIFAR10_DVS_SENSOR_SIZE),
            RandomFlipLR(p=config.flip_lr_p, sensor_width=width, sensor_size=CIFAR10_DVS_SENSOR_SIZE),
        ]
    )


def build_loader_kwargs(batch_size, generator, num_workers, drop_last):
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=seed_worker,
        drop_last=drop_last,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return loader_kwargs


def train(model, train_loader, optimizer, criterion, device):
    """
    one epoch training through the whole train dataset
    """
    model.train()
    tot_loss = 0.0
    tot_right = 0
    tot_samples = 0

    for x, y in train_loader:
        if os.getenv("DIFFLOGIC_DEBUG_SHAPES") == "1" and not hasattr(train, "_debug_batch_printed"):
            print("train batch input:", {"x_shape": tuple(x.shape), "y_shape": tuple(y.shape)})
            train._debug_batch_printed = True
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        batch_loss = criterion(logits, y)
        batch_loss.backward()
        optimizer.step()
        
        batch_size = y.shape[0] 
        tot_loss += batch_loss.item() * batch_size
        tot_right += (logits.argmax(dim=1) == y).sum().item()
        tot_samples += batch_size

    return {
        "loss": tot_loss / tot_samples,
        "accuracy": tot_right / tot_samples,
    }


def evaluate(model, val_loader, criterion, device):
    model.eval()
    tot_loss = 0.0
    tot_right = 0
    tot_samples = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            logits = forward_with_eval_batch_padding(model, x)
            batch_loss = criterion(logits, y)

            batch_size = y.shape[0]
            tot_loss += batch_loss.item() * batch_size
            tot_right += (logits.argmax(dim=1) == y).sum().item()
            tot_samples += batch_size

    return {
        "loss": tot_loss / tot_samples,
        "accuracy": tot_right / tot_samples,
    }


def collect_predictions(model, data_loader, device):
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = forward_with_eval_batch_padding(model, x)
            all_predictions.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(y.cpu().numpy())

    return {
        "targets": np.concatenate(all_targets),
        "predictions": np.concatenate(all_predictions),
    }


def forward_with_eval_batch_padding(model, x, batch_multiple=CONV_LOGIC_EVAL_BATCH_MULTIPLE):
    batch_size = x.shape[0]
    remainder = batch_size % batch_multiple
    if remainder == 0:
        return model(x)

    pad_count = batch_multiple - remainder
    pad_indices = torch.arange(pad_count, device=x.device) % batch_size
    x_padded = torch.cat((x, x.index_select(0, pad_indices)), dim=0)
    logits = model(x_padded)
    return logits[:batch_size]


def build_confusion_matrix_heatmap(y_true, y_pred, class_names, title, normalize=False):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    labels = list(range(len(class_names)))
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true" if normalize else None,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    text_threshold = float(matrix.max()) * 0.5 if matrix.size > 0 else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text = f"{value:.2f}" if normalize else str(int(value))
            text_color = "white" if value > text_threshold else "black"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color=text_color, fontsize=9)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    wandb_image = wandb.Image(fig)
    plt.close(fig)
    return wandb_image


def save_model(model, ckpt_path):
    ct = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{model.__class__.__name__}_{ct}.pth"
    location = os.path.join(ckpt_path, label)
    torch.save(model.state_dict(), location)
    return label[:-4], location


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name for wandb")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--model_scale", type=str, default="s", choices=["s", "m", "b", "l", "g"], help="Model scale")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--debug_shapes", action="store_true", help="Print one-time tensor shapes during the first forward pass")
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
        default=0.0,
        help="Minimum learning rate for cosine scheduler",
    )
    parser.add_argument(
        "--weights_init_mode",
        type=str,
        default="residual",
        choices=["residual", "gaussian"],
        help="Gate weight initialization for every logic layer.",
    )
    parser.add_argument("--num_time_bins", type=int, default=5, help="Number of temporal bins for the spike tensor")
    parser.add_argument(
        "--binning_strategy",
        type=str,
        default="duration",
        choices=("duration", "event_count"),
        help="How to assign events to temporal bins.",
    )
    parser.add_argument("--target_size", type=int, default=64, help="Square spatial size fed into LogicTreeNet")
    parser.add_argument("--spatial_jitter_max_shift", type=int, default=1, help="Maximum event jitter in pixels")
    parser.add_argument("--flip_lr_p", type=float, default=0.5, help="Horizontal flip probability for train samples")
    parser.add_argument(
        "--denoise_filter_time_us",
        type=float,
        default=50_000.0,
        help="Filter time passed to tonic.transforms.Denoise before train/val/test processing.",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Load spike tensors from pre-computed cache if available",
    )
    parser.add_argument(
        "--delete_cache_after_training",
        action="store_true",
        help="Delete cache directory after training completes",
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
    args = parser.parse_args()

    load_dotenv()
    if args.debug_shapes:
        os.environ["DIFFLOGIC_DEBUG_SHAPES"] = "1"
    wandb.login()

    SEED = 15
    OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "./outputs"))
    DATA_PATH = Path(os.getenv("DATA_PATH", "./data"))
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("device = {}".format(DEVICE))
    print("data path = {}".format(DATA_PATH))

    model_config = dict(
        # group sum 
        tau_gs = args.tau_gs,
        lr_tau_gs = 0,
        
        # gumble noise
        tau_noise= 0,
        
        # generic configs
        lr_model = 0.02,
        scheduler=args.scheduler,
        lr_min=args.lr_min,
        
        # training configs
        weight_decay = 0.002, 
        batch_size=args.batch_size,
    ) 

    base_config = dict(
        epochs=args.epochs,
        train_size=0.9,
        dataset=(
            f"CIFAR10-DVS-spike_tensor-{args.num_time_bins}bin"
            f"{'-event-count-binning' if args.binning_strategy == 'event_count' else ''}"
        ),
        architecture="LogicTreeNet",
        model_scale=args.model_scale,
        seed=SEED,
        num_time_bins=args.num_time_bins,
        binning_strategy=args.binning_strategy,
        target_size=args.target_size,
        spatial_jitter_max_shift=args.spatial_jitter_max_shift,
        flip_lr_p=args.flip_lr_p,
        denoise_filter_time_us=args.denoise_filter_time_us,
        num_workers=DEFAULT_NUM_WORKERS,
        weights_init_mode=args.weights_init_mode,
        affine_degrees=args.affine_degrees,
        affine_translate=args.affine_translate,
        affine_scale=args.affine_scale,
        erase_p=args.erase_p,
    )
    
    config = {**base_config, **model_config}
    
    # ---------- reproducibility -----------
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # --------------------------------------

    with wandb.init(project=os.getenv("WANDB_PROJECT", "CIFAR-10-DVS"), config=config, mode="online", name=args.run_name) as run:
        # access all HPs through wandb.config, so logging matches execution
        config = wandb.config 
        
        ckpt_path, config_path, _ = create_output_dirs(os.path.join(OUTPUT_PATH, run.name)) 
        json.dump(dict(config), open(os.path.join(config_path,"config.json"), "w"), indent=4)
        
        model = LogicTreeNet(
            config.model_scale,
            in_ch=2 * config.num_time_bins,
            out_classes=10,
            tau_gs=config.tau_gs,
            tau_noise=config.tau_noise,
            learn_tau_gs=config.lr_tau_gs == 0,
            input_size=config.target_size,
            weights_init_mode=config.weights_init_mode,
        )
        criterion = nn.CrossEntropyLoss()
        download_cifar10_dvs(DATA_PATH)

        # Only create denoiser if not using cache (since denoise is done during cache generation)
        if args.use_cache:
            event_filter = None
        else:
            event_filter = Denoise(filter_time=config.denoise_filter_time_us)
        
        split_indices = build_cifar10_dvs_splits(DATA_PATH, train_size=config.train_size, seed=SEED)
        dataset_kwargs = dict(
            data_path=DATA_PATH,
            representation="spike_tensor",
            target_size=(config.target_size, config.target_size),
            num_time_bins=config.num_time_bins,
            event_filter=event_filter,
            binning_strategy=config.binning_strategy,
            use_cache=args.use_cache,
            denoise_filter_time_us=config.denoise_filter_time_us,
            canonicalize_orientation=True,
        )
        train_dataset = CIFAR10DVS(
            indices=split_indices["train"],
            event_transform=build_train_event_transform(config),
            transform=build_binary_augmentation(config.affine_degrees, config.affine_translate, config.affine_scale, config.erase_p),
            **dataset_kwargs,
        )
        val_dataset = CIFAR10DVS(indices=split_indices["val"], event_transform=None, **dataset_kwargs)
        test_dataset = CIFAR10DVS(indices=split_indices["test"], event_transform=None, **dataset_kwargs)

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **build_loader_kwargs(config.batch_size, g, config.num_workers, True),
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **build_loader_kwargs(config.batch_size, g, config.num_workers, False),
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **build_loader_kwargs(config.batch_size, g, config.num_workers, False),
        )

        # separate parameters  
        if config.lr_tau_gs == 0:
            # All parameters except tau
            other_params = [p for n, p in model.named_parameters() if not n.endswith("group_sum.tau")]
            # Tau parameter
            tau_params = [p for n, p in model.named_parameters() if n.endswith("group_sum.tau")]
            # Define optimizer with different learning rates
            params_list = [
                {"params": other_params, "lr": config.lr_model},
                {"params": tau_params, "lr": config.lr_tau_gs}
            ]
        else:
            # If tau is not learnable, use a single learning rate for all
            params_list = [{"params": model.parameters(), "lr": config.lr_model}]
        optimizer = torch.optim.AdamW(params_list, weight_decay=config.weight_decay)
        scheduler = None
        if config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.epochs,
                eta_min=config.lr_min,
            )

        model.to(DEVICE)   
        
        wandb.watch(model, log="all", log_freq=1000)

        train_loss = []
        val_loss = []
        val_accuracy = []
        best_accuracy = 0.0
        best_epoch = 0
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        pbar = tqdm(range(config.epochs), desc="training epochs")
        for epoch in pbar: 
            train_metrics = train(
                model,
                train_loader,
                optimizer,
                criterion,
                DEVICE,
            )
            train_loss.append(train_metrics["loss"])

            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                DEVICE,
            )
            val_loss.append(val_metrics["loss"])
            val_accuracy.append(val_metrics["accuracy"])

            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                best_epoch = epoch
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            wandb.log({"epoch": epoch+1, 
                       "train_loss": train_metrics["loss"],
                       "val_loss": val_metrics["loss"],
                       "train_accuracy": train_metrics["accuracy"],
                       "val_accuracy": val_metrics["accuracy"],
                       "lr_model": optimizer.param_groups[0]["lr"],
                       "tau_gs": model.group_sum.tau.detach().item(), 
                       })
            if scheduler is not None:
                scheduler.step()
            
            pbar.set_postfix({"train loss (curr, prev)":
                            (f"{train_metrics['loss']:.3f}", f"{train_loss[-2]:.3f}" if len(train_loss) > 1 else None),
                            "val loss":f"{val_metrics['loss']:.3f}", 
                            "train acc":f"{train_metrics['accuracy']:.3f}", 
                            "val acc":f"{val_metrics['accuracy']:.3f}", })
        
        print("training finished")

        model.load_state_dict(best_state_dict)
        print("restored best weights from epoch {}".format(best_epoch+1))

        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            DEVICE,
        )
        test_predictions = collect_predictions(model, test_loader, DEVICE)
        wandb.log({"best_epoch": best_epoch+1, 
                   "best_val_loss": val_loss[best_epoch],
                   "best_val_accuracy": val_accuracy[best_epoch],
                   "test_loss": test_metrics["loss"],
                   "test_accuracy": test_metrics["accuracy"],
                   "test_confusion_matrix_counts": build_confusion_matrix_heatmap(
                       test_predictions["targets"],
                       test_predictions["predictions"],
                       CIFAR10_DVS_CLASS_NAMES,
                       title="Final Test Confusion Matrix",
                       normalize=False,
                   ),
                   "test_confusion_matrix_normalized": build_confusion_matrix_heatmap(
                       test_predictions["targets"],
                       test_predictions["predictions"],
                       CIFAR10_DVS_CLASS_NAMES,
                       title="Final Test Confusion Matrix (Normalized)",
                       normalize=True,
                   )})

        label, path_saved = save_model(model, ckpt_path)
        print("model saved to {}".format(path_saved))

        # Delete cache if requested - only now, after the test evaluation above,
        # which reads through the same cached datasets and would rewrite it.
        if args.delete_cache_after_training and args.use_cache:
            cache_dir = Path(DATA_PATH) / "CIFAR10-DVS-cache"
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                print(f"Cache directory deleted: {cache_dir}")

        model_artifact = wandb.Artifact(label, type="model", metadata=dict(config))
        model_artifact.add_file(path_saved)
        wandb.save(path_saved, base_path=ckpt_path)
        wandb.log_artifact(model_artifact)
        print("wandb log completed\n------------------------")        

if __name__ == "__main__":
    main()
