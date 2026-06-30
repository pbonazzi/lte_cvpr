import os
from datetime import datetime
from math import ceil

import numpy as np
import torch
import wandb
from sklearn.metrics import confusion_matrix
from torch.utils.data import BatchSampler, Dataset

from data.datasets.ncaltech101 import select_top_k_categories


class FilteredNCaltech101(Dataset):
    def __init__(self, base_dataset, selected_labels, class_names):
        self.base_dataset = base_dataset
        self.selected_labels = list(selected_labels)
        self.categories = list(class_names)
        self.label_map = {label: idx for idx, label in enumerate(self.selected_labels)}

        base_labels = base_dataset.labels.cpu().numpy()
        selected_mask = np.isin(base_labels, self.selected_labels)
        self.indices = np.flatnonzero(selected_mask).tolist()
        self.labels = torch.tensor(
            [self.label_map[int(base_labels[idx])] for idx in self.indices],
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        frame, label = self.base_dataset[self.indices[index]]
        mapped_label = self.label_map[int(label)]
        return frame, torch.tensor(mapped_label, dtype=torch.long)


class ClassBalancedBatchSampler(BatchSampler):
    """
    Build batches that cover as many classes as possible before repeating classes.
    When batch_size >= num_classes, each batch contains at least one sample per class.
    """

    def __init__(self, labels, batch_size, drop_last=True, seed=0):
        self.labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        if self.labels.ndim != 1:
            raise ValueError(f"Expected 1D labels, got shape {self.labels.shape}.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")
        if len(self.labels) == 0:
            raise ValueError("ClassBalancedBatchSampler requires a non-empty dataset.")

        unique_labels = np.unique(self.labels)
        self.class_labels = unique_labels.astype(np.int64, copy=False)
        self.class_indices = {
            int(label): np.flatnonzero(self.labels == label).astype(np.int64, copy=False)
            for label in self.class_labels
        }

    def __len__(self):
        if self.drop_last:
            return len(self.labels) // self.batch_size
        return ceil(len(self.labels) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        per_class_indices = {
            label: rng.permutation(indices).tolist()
            for label, indices in self.class_indices.items()
        }
        class_offsets = {label: 0 for label in self.class_indices}

        total_batches = len(self)
        total_items = total_batches * self.batch_size if self.drop_last else len(self.labels)

        emitted = 0
        while emitted < total_items:
            current_batch_size = min(self.batch_size, total_items - emitted)
            unique_class_count = min(current_batch_size, len(self.class_labels))

            chosen_classes = rng.permutation(self.class_labels)[:unique_class_count].tolist()
            while len(chosen_classes) < current_batch_size:
                chosen_classes.append(int(rng.choice(self.class_labels)))

            batch_indices = []
            for class_label in chosen_classes:
                class_label = int(class_label)
                indices = per_class_indices[class_label]
                offset = class_offsets[class_label]
                if offset >= len(indices):
                    indices = rng.permutation(self.class_indices[class_label]).tolist()
                    per_class_indices[class_label] = indices
                    offset = 0

                batch_indices.append(int(indices[offset]))
                class_offsets[class_label] = offset + 1

            rng.shuffle(batch_indices)
            yield batch_indices
            emitted += len(batch_indices)


def select_top_k_labels(root_path, categories, top_k):
    del categories
    return select_top_k_categories(root_path, top_k)


def build_class_weights(class_counts, device):
    counts = torch.tensor(class_counts, dtype=torch.float32, device=device)
    weights = counts.sum() / (counts.numel() * counts.clamp_min(1.0))
    return weights


def select_sample_indices(labels, num_samples, strategy, seed):
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    total_samples = len(labels)
    if total_samples == 0 or num_samples <= 0:
        return []

    num_samples = min(num_samples, total_samples)
    rng = np.random.default_rng(seed)

    if strategy == "random":
        return rng.choice(total_samples, size=num_samples, replace=False).tolist()

    unique_labels, counts = np.unique(labels, return_counts=True)
    class_indices = {
        label: rng.permutation(np.flatnonzero(labels == label)).tolist()
        for label in unique_labels
    }

    if strategy == "proportional":
        expected = counts / counts.sum() * num_samples
        quotas = np.floor(expected).astype(int)
        remainder = num_samples - quotas.sum()

        if remainder > 0:
            fractional = expected - quotas
            for idx in np.argsort(fractional)[::-1][:remainder]:
                quotas[idx] += 1

        selected = []
        for label, quota in zip(unique_labels, quotas):
            selected.extend(class_indices[label][:quota])

        if len(selected) < num_samples:
            remaining = list(set(range(total_samples)) - set(selected))
            extra = rng.choice(remaining, size=num_samples - len(selected), replace=False).tolist()
            selected.extend(extra)

        return selected

    if strategy != "stratified":
        raise ValueError(f"Unsupported sample strategy: {strategy}")

    class_order = rng.permutation(unique_labels)
    class_offsets = {label: 0 for label in unique_labels}
    selected = []

    while len(selected) < num_samples:
        progressed = False
        for label in class_order:
            offset = class_offsets[label]
            candidates = class_indices[label]
            if offset >= len(candidates):
                continue
            selected.append(candidates[offset])
            class_offsets[label] += 1
            progressed = True
            if len(selected) == num_samples:
                break
        if not progressed:
            break

    return selected


def frame_to_rgb_image(frame):
    frame = frame.detach().cpu().float().clamp(min=0)
    if frame.ndim != 3:
        raise ValueError(f"Expected frame with shape [C, H, W], got {tuple(frame.shape)}")

    if frame.shape[0] == 2:
        vis_frame = frame
    elif frame.shape[0] > 2 and frame.shape[0] % 2 == 0:
        vis_frame = frame.view(frame.shape[0] // 2, 2, frame.shape[1], frame.shape[2]).sum(dim=0)
    else:
        vis_frame = frame[:2]

    if vis_frame.shape[0] == 2:
        pos_mask = vis_frame[0].numpy() > 0
        neg_mask = vis_frame[1].numpy() > 0
        overlap_mask = pos_mask & neg_mask
        image = np.ones((vis_frame.shape[1], vis_frame.shape[2], 3), dtype=np.float32)
        image[pos_mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        image[neg_mask & ~pos_mask] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        image[overlap_mask] = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    else:
        max_value = float(vis_frame.max().item())
        if max_value <= 0:
            max_value = 1.0
        image = (vis_frame[:3] / max_value).permute(1, 2, 0).numpy()

    return image


def frame_to_wandb_image(frame):
    return wandb.Image(frame_to_rgb_image(frame))


def build_class_count_table(labels, class_names):
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    counts = np.bincount(labels, minlength=len(class_names))
    table = wandb.Table(columns=["class_idx", "class_name", "count"])
    for class_idx, class_name in enumerate(class_names):
        table.add_data(class_idx, class_name, int(counts[class_idx]))
    return table


def collect_prediction_samples(model, dataset, sample_indices, class_names, device, split_name):
    if not sample_indices:
        return []

    frames = []
    labels = []
    for sample_idx in sample_indices:
        frame, label = dataset[sample_idx]
        frames.append(frame)
        labels.append(int(label))

    batch = torch.stack(frames).to(device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        predictions = model(batch).argmax(dim=1).cpu().tolist()
    if was_training:
        model.train()

    records = []
    for sample_idx, frame, label, pred in zip(sample_indices, frames, labels, predictions):
        records.append(
            {
                "split": split_name,
                "sample_idx": int(sample_idx),
                "ground_truth": class_names[label],
                "predicted": class_names[pred],
                "frame": frame.detach().cpu(),
            }
        )

    return records


def build_prediction_table(records):
    if not records:
        return None

    table = wandb.Table(columns=["split", "sample_idx", "ground_truth", "predicted", "image"])
    for record in records:
        table.add_data(
            record["split"],
            record["sample_idx"],
            record["ground_truth"],
            record["predicted"],
            frame_to_wandb_image(record["frame"]),
        )

    return table


def train_classifier_epoch(model, train_loader, optimizer, criterion, device):
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


def evaluate_classifier(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_right = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            batch_loss = criterion(logits, y)

            batch_size = y.shape[0]
            total_loss += batch_loss.item() * batch_size
            total_right += (logits.argmax(dim=1) == y).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_right / total_samples


def collect_classifier_predictions(model, data_loader, device, batch_unpack_fn=None):
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in data_loader:
            if batch_unpack_fn is None:
                x, y = batch[:2]
            else:
                x, y = batch_unpack_fn(batch)

            x = x.to(device)
            logits = model(x)
            all_predictions.append(logits.argmax(dim=1).cpu().numpy())
            all_targets.append(y.cpu().numpy())

    return np.concatenate(all_targets), np.concatenate(all_predictions)


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

    fig_width = max(6, 0.8 * len(class_names) + 2)
    fig_height = max(5, 0.8 * len(class_names) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    text_threshold = float(matrix.max()) * 0.5 if matrix.size > 0 else 0.0
    annotate = len(class_names) <= 12
    if annotate:
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
