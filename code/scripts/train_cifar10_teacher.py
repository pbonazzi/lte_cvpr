"""Train a real-valued teacher (CIFAR-style ResNet-18) on the same binary CIFAR10-DVS spike tensors,
splits and augmentation as the LogicTreeNet trainer, for knowledge distillation into a logic network.

Saves the best-validation weights plus the config needed to rebuild it (see load_teacher).
"""
import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from data.datasets.cifar10_dvs import CIFAR10DVS, build_cifar10_dvs_splits
from data.transforms import build_binary_augmentation
from data.utils import seed_worker
from scripts.train_cifar10_spike_tensor import build_train_event_transform

def build_teacher(in_ch, num_classes=10):
    """ResNet-18 with a 3x3 stride-1 stem and no max-pool, as usual for small images."""
    model = torchvision.models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def load_teacher(path, device="cuda"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_teacher(ckpt["in_ch"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["config"]


def accuracy(model, loader, device):
    model.eval()
    right = total = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y in loader:
            right += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
            total += len(y)
    return right / total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target_size", type=int, default=64)
    p.add_argument("--num_time_bins", type=int, default=5)
    p.add_argument("--pool_thresholds", type=str, default="", help="as in train_cifar10_spike_tensor.py")
    p.add_argument("--spatial_jitter_max_shift", type=int, default=8)
    p.add_argument("--flip_lr_p", type=float, default=0.5)
    p.add_argument("--affine_degrees", type=float, default=15.0)
    p.add_argument("--affine_translate", type=float, default=0.0)
    p.add_argument("--affine_scale", type=float, default=0.1)
    p.add_argument("--erase_p", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--use_cache", action="store_true")
    p.add_argument("--seed", type=int, default=15)
    p.add_argument("--run_name", type=str, default=None)
    args = p.parse_args()
    load_dotenv()
    DATA_PATH, OUTPUT_PATH = os.getenv("DATA_PATH", "data"), os.getenv("OUTPUT_PATH", "outputs")

    config = dict(vars(args), denoise_filter_time_us=50_000.0, binning_strategy="duration", train_size=0.9,
                  pool_thresholds=[int(v) for v in args.pool_thresholds.split(",") if v])
    in_ch = 2 * args.num_time_bins * max(1, len(config["pool_thresholds"]))
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    device = "cuda"

    splits = build_cifar10_dvs_splits(DATA_PATH, train_size=0.9, seed=15)   # same fixed split as the students
    kw = dict(data_path=DATA_PATH, representation="spike_tensor", target_size=(args.target_size, args.target_size),
              num_time_bins=args.num_time_bins, event_filter=None, binning_strategy="duration",
              use_cache=args.use_cache, denoise_filter_time_us=50_000.0, canonicalize_orientation=True,
              pool_thresholds=config["pool_thresholds"])
    train_ds = CIFAR10DVS(indices=splits["train"], event_transform=build_train_event_transform(SimpleCfg(args)),
                          transform=build_binary_augmentation(args.affine_degrees, args.affine_translate,
                                                              args.affine_scale, args.erase_p), **kw)
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=shuffle,
                                            num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g,
                                            pin_memory=True, persistent_workers=args.num_workers > 0)
    train_loader = loader(train_ds, True)
    val_loader = loader(CIFAR10DVS(indices=splits["val"], event_transform=None, **kw), False)
    test_loader = loader(CIFAR10DVS(indices=splits["test"], event_transform=None, **kw), False)

    model = build_teacher(in_ch).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    name = args.run_name or f"cifar10dvs-teacher-resnet18-{datetime.now():%Y%m%d-%H%M}"
    out_dir = os.path.join(OUTPUT_PATH, name)
    os.makedirs(out_dir, exist_ok=True)
    best_val, best_state, best_epoch = -1.0, None, -1
    with wandb.init(project=os.getenv("WANDB_PROJECT", "lte-cvpr"), name=name, config=config) as run:
        for epoch in range(args.epochs):
            model.train()
            right = total = 0
            for x, y in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                right += (logits.argmax(1) == y).sum().item()
                total += len(y)
            sched.step()
            val_acc = accuracy(model, val_loader, device)
            if val_acc > best_val:
                best_val, best_epoch = val_acc, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            run.log({"epoch": epoch + 1, "train_accuracy": right / total, "val_accuracy": val_acc, "loss": loss.item()})
            print(f"epoch {epoch + 1} train {right / total:.4f} val {val_acc:.4f}", flush=True)
        model.load_state_dict(best_state)
        test_acc = accuracy(model, test_loader, device)
        torch.save({"state_dict": best_state, "in_ch": in_ch, "config": config}, os.path.join(out_dir, "teacher.pth"))
        json.dump(config, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
        run.summary.update({"best_val_accuracy": best_val, "best_epoch": best_epoch, "test_accuracy": test_acc})
        print(f"best val {best_val:.4f} at epoch {best_epoch + 1}, test {test_acc:.4f}, saved {out_dir}/teacher.pth")


class SimpleCfg:
    """The two fields build_train_event_transform reads."""
    def __init__(self, args):
        self.spatial_jitter_max_shift = args.spatial_jitter_max_shift
        self.flip_lr_p = args.flip_lr_p


if __name__ == "__main__":
    main()
