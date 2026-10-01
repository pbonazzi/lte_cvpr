import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.datasets.cifar10_dvs import (
    CIFAR10DVS,
    CIFAR10_DVS_CLASS_NAMES,
    CIFAR10_DVS_SENSOR_SIZE,
    build_cifar10_dvs_splits,
    download_cifar10_dvs,
)
from data.transforms import Denoise, EventTransformCompose, RandomFlipLR, SpatialJitter
from data.utils import create_output_dirs, seed_worker
from models.logictreenet import LogicTreeNet

CONV_LOGIC_EVAL_BATCH_MULTIPLE = 16
SEED = 15
DEFAULT_NUM_WORKERS = 8
EMBEDDING_PLOT_POINTS = 3000


def build_train_event_transform(config):
    _height, width = CIFAR10_DVS_SENSOR_SIZE
    return EventTransformCompose(
        [
            SpatialJitter(max_shift=config.spatial_jitter_max_shift, sensor_size=CIFAR10_DVS_SENSOR_SIZE),
            RandomFlipLR(p=config.flip_lr_p, sensor_width=width, sensor_size=CIFAR10_DVS_SENSOR_SIZE),
        ]
    )


def build_loader_kwargs(config, generator, drop_last):
    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        generator=generator,
        worker_init_fn=seed_worker,
        drop_last=drop_last,
        pin_memory=True,
    )
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return loader_kwargs


class AuxProxyBranch(nn.Module):
    def __init__(self, in_channels, projection_dim, num_classes):
        super().__init__()
        # Pool spatial activations so the auxiliary head works on one vector
        # per sample instead of a flattened feature map.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection_head = nn.Sequential(
            nn.Linear(in_channels, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        # One learnable proxy per class; embeddings are pulled toward their
        # class proxy and pushed away from the other class proxies.
        self.proxies = nn.Parameter(torch.randn(num_classes, projection_dim))

    def forward(self, x):
        pooled = self.pool(x).flatten(start_dim=1)
        return F.normalize(self.projection_head(pooled), dim=1)


class BlockAuxProxyLogicTreeNet(nn.Module):
    def __init__(self, base_model, projection_dim, num_classes, use_block3_aux=True, use_block4_aux=True):
        super().__init__()
        self.base_model = base_model

        block3_channels = self.base_model.net[2].net[0].out_ch
        block4_channels = self.base_model.net[3].net[0].out_ch

        self.block3_aux = (
            AuxProxyBranch(block3_channels, projection_dim, num_classes) if use_block3_aux else None
        )
        self.block4_aux = (
            AuxProxyBranch(block4_channels, projection_dim, num_classes) if use_block4_aux else None
        )

    def forward(self, x):
        x = self.base_model.net[0](x)
        x = self.base_model.net[1](x)
        x = self.base_model.net[2](x)
        # Auxiliary supervision is attached after block 3 and block 4, while
        # the main classification path continues unchanged.
        block3_activation = x
        x = self.base_model.net[3](x)
        block4_activation = x
        x = self.base_model.net[4](x)
        x = self.base_model.net[5](x)
        x = self.base_model.net[6](x)
        features = self.base_model.net[7](x)

        logits_input = features
        if self.base_model.pad_features > 0:
            logits_input = F.pad(logits_input, (0, self.base_model.pad_features))
        logits = self.base_model.group_sum(logits_input)

        outputs = {
            "logits": logits,
            "features": features,
        }
        if self.block3_aux is not None:
            outputs["block3_embeddings"] = self.block3_aux(block3_activation)
        if self.block4_aux is not None:
            outputs["block4_embeddings"] = self.block4_aux(block4_activation)
        return outputs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name for wandb")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--model_scale", type=str, default="s", choices=["s", "m", "b", "l", "g"], help="Model scale")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--debug_shapes", action="store_true", help="Print one-time tensor shapes during the first forward pass")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"], help="Learning rate scheduler")
    parser.add_argument("--lr_model", type=float, default=0.02, help="Learning rate for non-tau parameters")
    parser.add_argument("--lr_min", type=float, default=0.0, help="Minimum learning rate for cosine scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.002, help="AdamW weight decay")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--projection_dim", type=int, default=128, help="Embedding dimension for block3/block4 proxy branches")
    parser.add_argument("--block3_metric_weight", type=float, default=0.1, help="Weight for block3 auxiliary proxy anchor loss")
    parser.add_argument("--block4_metric_weight", type=float, default=0.1, help="Weight for block4 auxiliary proxy anchor loss")
    parser.add_argument("--proxy_logit_scale", type=float, default=20.0, help="Scale applied to cosine similarities")
    parser.add_argument("--proxy_anchor_margin", type=float, default=0.3, help="Margin used in proxy anchor loss")
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
    return parser.parse_args()


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


def build_run_config(args):
    return {
        "epochs": args.epochs,
        "train_size": 0.9,
        "dataset": (
            f"CIFAR10-DVS-spike_tensor-{args.num_time_bins}bin"
            f"{'-event-count-binning' if args.binning_strategy == 'event_count' else ''}"
            f"-block34-aux-proxy-anchor"
        ),
        "architecture": "LogicTreeNet+Block3Block4AuxProxyAnchor",
        "model_scale": args.model_scale,
        "seed": SEED,
        "num_time_bins": args.num_time_bins,
        "binning_strategy": args.binning_strategy,
        "target_size": args.target_size,
        "spatial_jitter_max_shift": args.spatial_jitter_max_shift,
        "flip_lr_p": args.flip_lr_p,
        "denoise_filter_time_us": args.denoise_filter_time_us,
        "use_cache": args.use_cache,
        "tau_gs": 20,
        "lr_tau_gs": 0,
        "tau_noise": 0,
        "lr_model": args.lr_model,
        "scheduler": args.scheduler,
        "lr_min": args.lr_min,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "num_workers": DEFAULT_NUM_WORKERS,
        "projection_dim": args.projection_dim,
        "block3_metric_weight": args.block3_metric_weight,
        "block4_metric_weight": args.block4_metric_weight,
        "proxy_logit_scale": args.proxy_logit_scale,
        "proxy_anchor_margin": args.proxy_anchor_margin,
    }


def proxy_anchor_loss(embeddings, labels, proxies, logit_scale, margin=0.1):
    # Positive terms pull embeddings toward their class proxy; negative terms
    # push embeddings away from the other class proxies.
    normalized_embeddings = F.normalize(embeddings, dim=1)
    normalized_proxies = F.normalize(proxies, dim=1)

    similarity = torch.matmul(normalized_embeddings, normalized_proxies.T)
    positive_mask = F.one_hot(labels, num_classes=similarity.shape[1]).bool()
    negative_mask = ~positive_mask

    positive_terms = torch.exp(-logit_scale * (similarity - margin)) * positive_mask.float()
    positive_proxy_sums = positive_terms.sum(dim=0)
    valid_positive_proxies = positive_mask.any(dim=0)
    if valid_positive_proxies.any():
        positive_loss = torch.log1p(positive_proxy_sums[valid_positive_proxies]).sum()
        positive_loss = positive_loss / valid_positive_proxies.sum().clamp_min(1)
    else:
        positive_loss = similarity.new_zeros(())

    negative_terms = torch.exp(logit_scale * (similarity + margin)) * negative_mask.float()
    negative_proxy_sums = negative_terms.sum(dim=0)
    negative_loss = torch.log1p(negative_proxy_sums).sum() / similarity.shape[1]
    loss = positive_loss + negative_loss

    positive_similarity = similarity.gather(1, labels.unsqueeze(1)).squeeze(1)
    proxy_logits = similarity * logit_scale
    accuracy = (proxy_logits.argmax(dim=1) == labels).float().mean().item()

    if similarity.shape[1] > 1:
        negative_similarity = similarity.masked_fill(positive_mask, float("-inf"))
        hardest_negative = negative_similarity.max(dim=1).values
        mean_hardest_negative = hardest_negative.mean().item()
    else:
        mean_hardest_negative = 0.0

    stats = {
        "target_proxy_similarity": positive_similarity.mean().item(),
        "hardest_negative_similarity": mean_hardest_negative,
        "proxy_accuracy": accuracy,
    }
    return loss, stats


def average_metric_stats(total_metric_stats, total_batches):
    return {key: value / max(total_batches, 1) for key, value in total_metric_stats.items()}


def prefix_metric_stats(prefix, metric_stats):
    return {f"{prefix}_{key}": value for key, value in metric_stats.items()}


def slice_model_output(model_output, batch_size):
    return {name: value[:batch_size] for name, value in model_output.items()}


def forward_with_eval_batch_padding(model, x, batch_multiple=CONV_LOGIC_EVAL_BATCH_MULTIPLE):
    batch_size = x.shape[0]
    remainder = batch_size % batch_multiple
    if remainder == 0:
        return model(x)

    pad_count = batch_multiple - remainder
    pad_indices = torch.arange(pad_count, device=x.device) % batch_size
    x_padded = torch.cat((x, x.index_select(0, pad_indices)), dim=0)
    return slice_model_output(model(x_padded), batch_size)


def compute_aux_metric_loss(config, model, outputs, labels):
    # The auxiliary metric losses are added to cross-entropy during training
    # only; inference uses the main LogicTreeNet logits.
    total_aux_loss = outputs["logits"].new_zeros(())
    branch_losses = {
        "block3_metric_loss": 0.0,
        "block4_metric_loss": 0.0,
    }
    combined_stats = {}

    if "block3_embeddings" in outputs and config.block3_metric_weight > 0:
        block3_loss, block3_stats = proxy_anchor_loss(
            outputs["block3_embeddings"],
            labels,
            model.block3_aux.proxies,
            config.proxy_logit_scale,
            margin=config.proxy_anchor_margin,
        )
        total_aux_loss = total_aux_loss + config.block3_metric_weight * block3_loss
        branch_losses["block3_metric_loss"] = block3_loss.item()
        combined_stats.update(prefix_metric_stats("block3", block3_stats))

    if "block4_embeddings" in outputs and config.block4_metric_weight > 0:
        block4_loss, block4_stats = proxy_anchor_loss(
            outputs["block4_embeddings"],
            labels,
            model.block4_aux.proxies,
            config.proxy_logit_scale,
            margin=config.proxy_anchor_margin,
        )
        total_aux_loss = total_aux_loss + config.block4_metric_weight * block4_loss
        branch_losses["block4_metric_loss"] = block4_loss.item()
        combined_stats.update(prefix_metric_stats("block4", block4_stats))

    return total_aux_loss, branch_losses, combined_stats


def train_epoch(model, train_loader, optimizer, ce_criterion, config, device):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_aux = 0.0
    total_block3_metric = 0.0
    total_block4_metric = 0.0
    total_right = 0
    total_samples = 0
    total_metric_stats = {}
    total_batches = 0

    for x, y in train_loader:
        if os.getenv("DIFFLOGIC_DEBUG_SHAPES") == "1" and not hasattr(train_epoch, "_debug_batch_printed"):
            print("train batch input:", {"x_shape": tuple(x.shape), "y_shape": tuple(y.shape)})
            train_epoch._debug_batch_printed = True

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        ce_loss = ce_criterion(outputs["logits"], y)
        aux_loss, branch_losses, metric_stats = compute_aux_metric_loss(config, model, outputs, y)
        loss = ce_loss + aux_loss
        loss.backward()
        optimizer.step()

        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size
        total_ce += ce_loss.item() * batch_size
        total_aux += aux_loss.item() * batch_size
        total_block3_metric += branch_losses["block3_metric_loss"] * batch_size
        total_block4_metric += branch_losses["block4_metric_loss"] * batch_size
        total_right += (outputs["logits"].argmax(dim=1) == y).sum().item()
        total_samples += batch_size
        for key, value in metric_stats.items():
            total_metric_stats[key] = total_metric_stats.get(key, 0.0) + value
        total_batches += 1

    return {
        "loss": total_loss / total_samples,
        "ce_loss": total_ce / total_samples,
        "aux_loss": total_aux / total_samples,
        "block3_metric_loss": total_block3_metric / total_samples,
        "block4_metric_loss": total_block4_metric / total_samples,
        "accuracy": total_right / total_samples,
        "metric_stats": average_metric_stats(total_metric_stats, total_batches),
    }


def evaluate(model, data_loader, ce_criterion, config, device):
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_aux = 0.0
    total_block3_metric = 0.0
    total_block4_metric = 0.0
    total_right = 0
    total_samples = 0
    total_metric_stats = {}
    total_batches = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = forward_with_eval_batch_padding(model, x)
            ce_loss = ce_criterion(outputs["logits"], y)
            aux_loss, branch_losses, metric_stats = compute_aux_metric_loss(config, model, outputs, y)
            loss = ce_loss + aux_loss

            batch_size = y.shape[0]
            total_loss += loss.item() * batch_size
            total_ce += ce_loss.item() * batch_size
            total_aux += aux_loss.item() * batch_size
            total_block3_metric += branch_losses["block3_metric_loss"] * batch_size
            total_block4_metric += branch_losses["block4_metric_loss"] * batch_size
            total_right += (outputs["logits"].argmax(dim=1) == y).sum().item()
            total_samples += batch_size
            for key, value in metric_stats.items():
                total_metric_stats[key] = total_metric_stats.get(key, 0.0) + value
            total_batches += 1

    return {
        "loss": total_loss / total_samples,
        "ce_loss": total_ce / total_samples,
        "aux_loss": total_aux / total_samples,
        "block3_metric_loss": total_block3_metric / total_samples,
        "block4_metric_loss": total_block4_metric / total_samples,
        "accuracy": total_right / total_samples,
        "metric_stats": average_metric_stats(total_metric_stats, total_batches),
    }


def collect_predictions(model, data_loader, device):
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = forward_with_eval_batch_padding(model, x)
            all_predictions.append(outputs["logits"].argmax(dim=1).cpu().numpy())
            all_targets.append(y.cpu().numpy())

    return {
        "targets": np.concatenate(all_targets),
        "predictions": np.concatenate(all_predictions),
    }


def collect_feature_embeddings(model, data_loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            outputs = forward_with_eval_batch_padding(model, x)
            all_embeddings.append(outputs["features"].cpu().numpy().astype(np.float32, copy=False))
            all_labels.append(y.numpy())

    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)


def build_embedding_tsne_image(embeddings, labels, class_names, max_points, seed, title):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    sample_count = min(len(embeddings), max_points)
    rng = np.random.default_rng(seed)
    if sample_count < len(embeddings):
        indices = rng.choice(len(embeddings), size=sample_count, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    pca_dim = min(50, embeddings.shape[1], embeddings.shape[0])
    reduced_embeddings = PCA(n_components=pca_dim, random_state=seed).fit_transform(embeddings)
    perplexity = min(30, max(5, sample_count // 50))
    embedding_2d = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    ).fit_transform(reduced_embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")
    for class_idx, class_name in enumerate(class_names):
        class_mask = labels == class_idx
        if not np.any(class_mask):
            continue
        ax.scatter(
            embedding_2d[class_mask, 0],
            embedding_2d[class_mask, 1],
            s=10,
            alpha=0.7,
            color=cmap(class_idx),
            label=class_name,
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=2, fontsize=8, loc="best")
    fig.tight_layout()
    wandb_image = wandb.Image(fig)
    plt.close(fig)
    return wandb_image


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
    args = parse_args()

    load_dotenv()
    if args.debug_shapes:
        os.environ["DIFFLOGIC_DEBUG_SHAPES"] = "1"
    if args.wandb_mode != "disabled":
        wandb.login()

    output_path = Path(os.getenv("OUTPUT_PATH", "./outputs"))
    data_path = Path(os.getenv("DATA_PATH", "./data"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    print(f"data path = {data_path}")

    config = build_run_config(args)
    generator = seed_everything(SEED)

    with wandb.init(project=os.getenv("WANDB_PROJECT", "CIFAR-10-DVS"), config=config, mode=args.wandb_mode, name=args.run_name) as run:
        config = wandb.config

        ckpt_path, config_path, _ = create_output_dirs(os.path.join(output_path, run.name))
        with open(os.path.join(config_path, "config.json"), "w") as config_file:
            json.dump(dict(config), config_file, indent=4)

        base_model = LogicTreeNet(
            config.model_scale,
            in_ch=2 * config.num_time_bins,
            out_classes=10,
            tau_gs=config.tau_gs,
            tau_noise=config.tau_noise,
            learn_tau_gs=config.lr_tau_gs == 0,
            input_size=config.target_size,
        )
        model = BlockAuxProxyLogicTreeNet(
            base_model,
            config.projection_dim,
            num_classes=len(CIFAR10_DVS_CLASS_NAMES),
            use_block3_aux=config.block3_metric_weight > 0,
            use_block4_aux=config.block4_metric_weight > 0,
        )
        ce_criterion = nn.CrossEntropyLoss()

        download_cifar10_dvs(data_path)
        if args.use_cache:
            denoiser = None
        else:
            denoiser = Denoise(filter_time=config.denoise_filter_time_us)

        split_indices = build_cifar10_dvs_splits(data_path, train_size=config.train_size, seed=SEED)
        dataset_kwargs = dict(
            data_path=data_path,
            representation="spike_tensor",
            target_size=(config.target_size, config.target_size),
            num_time_bins=config.num_time_bins,
            event_filter=denoiser,
            binning_strategy=config.binning_strategy,
            use_cache=args.use_cache,
            denoise_filter_time_us=config.denoise_filter_time_us,
            canonicalize_orientation=True,
        )
        train_dataset = CIFAR10DVS(
            indices=split_indices["train"],
            event_transform=build_train_event_transform(config),
            **dataset_kwargs,
        )
        val_dataset = CIFAR10DVS(indices=split_indices["val"], event_transform=None, **dataset_kwargs)
        test_dataset = CIFAR10DVS(indices=split_indices["test"], event_transform=None, **dataset_kwargs)

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **build_loader_kwargs(config, generator, True),
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **build_loader_kwargs(config, generator, False),
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **build_loader_kwargs(config, generator, False),
        )

        if config.lr_tau_gs == 0:
            other_params = [p for name, p in model.named_parameters() if not name.endswith("group_sum.tau")]
            tau_params = [p for name, p in model.named_parameters() if name.endswith("group_sum.tau")]
            params = [
                {"params": other_params, "lr": config.lr_model},
                {"params": tau_params, "lr": config.lr_tau_gs},
            ]
        else:
            params = [{"params": model.parameters(), "lr": config.lr_model}]
        optimizer = torch.optim.AdamW(params, weight_decay=config.weight_decay)

        scheduler = None
        if config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.epochs,
                eta_min=config.lr_min,
            )

        model.to(device)

        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        best_epoch = 0
        best_accuracy = 0.0
        best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        progress_bar = tqdm(range(config.epochs), desc="training epochs")
        for epoch in progress_bar:
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                ce_criterion,
                config,
                device,
            )
            val_metrics = evaluate(
                model,
                val_loader,
                ce_criterion,
                config,
                device,
            )

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                best_epoch = epoch
                best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

            log_data = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_aux_loss": train_metrics["aux_loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_aux_loss": val_metrics["aux_loss"],
                "val_accuracy": val_metrics["accuracy"],
                "lr_model": optimizer.param_groups[0]["lr"],
                "tau_gs": model.base_model.group_sum.tau.detach().item(),
            }
            wandb.log(log_data)

            if scheduler is not None:
                scheduler.step()

            previous_train_loss = history["train_loss"][-2] if len(history["train_loss"]) > 1 else None
            progress_bar.set_postfix(
                {
                    "train": f"{train_metrics['loss']:.3f}",
                    "ce": f"{train_metrics['ce_loss']:.3f}",
                    "aux": f"{train_metrics['aux_loss']:.3f}",
                    "val": f"{val_metrics['loss']:.3f}",
                    "acc": f"{val_metrics['accuracy']:.3f}",
                    "prev": f"{previous_train_loss:.3f}" if previous_train_loss is not None else None,
                }
            )

        print("training finished")
        model.load_state_dict(best_state_dict)
        print(f"restored best weights from epoch {best_epoch + 1}")

        test_metrics = evaluate(
            model,
            test_loader,
            ce_criterion,
            config,
            device,
        )
        test_predictions = collect_predictions(model, test_loader, device)
        test_embeddings, test_embedding_labels = collect_feature_embeddings(model, test_loader, device)
        test_log_data = {
            "best_epoch": best_epoch + 1,
            "best_val_loss": history["val_loss"][best_epoch],
            "best_val_accuracy": history["val_accuracy"][best_epoch],
            "test_loss": test_metrics["loss"],
            "test_aux_loss": test_metrics["aux_loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_feature_tsne": build_embedding_tsne_image(
                test_embeddings,
                test_embedding_labels,
                CIFAR10_DVS_CLASS_NAMES,
                max_points=EMBEDDING_PLOT_POINTS,
                seed=SEED,
                title="Final Test Feature t-SNE",
            ),
            "test_confusion_matrix_normalized": build_confusion_matrix_heatmap(
                test_predictions["targets"],
                test_predictions["predictions"],
                CIFAR10_DVS_CLASS_NAMES,
                title="Final Test Confusion Matrix (Normalized)",
                normalize=True,
            ),
        }
        wandb.log(test_log_data)

        label, path_saved = save_model(model, ckpt_path)
        print(f"model saved to {path_saved}")

        if args.delete_cache_after_training and args.use_cache:
            cache_dir = Path(data_path) / "CIFAR10-DVS-cache"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                print(f"Cache directory deleted: {cache_dir}")

        model_artifact = wandb.Artifact(label, type="model", metadata=dict(config))
        model_artifact.add_file(path_saved)
        wandb.save(path_saved, base_path=ckpt_path)
        wandb.log_artifact(model_artifact)
        print("wandb log completed\n------------------------")


if __name__ == "__main__":
    main()
