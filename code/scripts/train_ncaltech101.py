import argparse
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv

import wandb
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from models.logictreenet import LogicTreeNet
from data.datasets.ncaltech101 import NCaltech101
from data.transforms import DATASET_TRANSFORM
from data.utils import seed_worker, create_output_dirs


def train(model, train_loader, optimizer, criterion, device):
    """
    one epoch training through the whole train dataset
    """
    model.train()
    tot_loss = 0.0
    tot_right = 0
    tot_samples = 0

    for x, y in train_loader:
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

    return tot_loss/tot_samples, tot_right/tot_samples


def evaluate(model, val_loader, criterion, device):
    model.eval()
    tot_loss = 0.0
    tot_right = 0
    tot_samples = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            batch_loss = criterion(logits, y)

            batch_size = y.shape[0]
            tot_loss += batch_loss.item() * batch_size
            tot_right += (logits.argmax(dim=1) == y).sum().item()
            tot_samples += batch_size

    return tot_loss/tot_samples, tot_right/tot_samples


def save_model(model, ckpt_path):
    ct = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{model.__class__.__name__}_{ct}.pth"
    location = os.path.join(ckpt_path, label)
    torch.save(model.state_dict(), location)
    return label[:-4], location


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None, help="Custom run name for wandb")
    args = parser.parse_args()

    load_dotenv()
    wandb.login()

    SEED = 15
    OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "./outputs"))
    DATA_PATH = Path(os.getenv("DATA_PATH", "./data"))

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("device = {}".format(DEVICE))
    print("data path = {}".format(DATA_PATH))

    model_config = dict(
        # group sum
        tau_gs = 20,
        lr_tau_gs = 0,

        # gumble noise
        tau_noise= 0,

        # generic configs
        lr_model = 0.02,

        # training configs
        weight_decay = 0.002,
        batch_size=16,
    )

    base_config = dict(
        epochs=200,
        train_size=0.8,
        dataset="N-Caltech101",
        architecture="LogicTreeNet",
        out_classes=101,
        model_scale="s",
        seed=SEED,
        target_size=64,
    )

    config = {**base_config, **model_config}

    # ---------- reproducibility -----------
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # --------------------------------------

    with wandb.init(project=os.getenv("WANDB_PROJECT", "LGN_Events"), config=config, mode="online", name=args.run_name) as run:
        # access all HPs through wandb.config, so logging matches execution
        config = wandb.config

        ckpt_path, config_path, log_path = create_output_dirs(os.path.join(OUTPUT_PATH, run.name))
        json.dump(dict(config), open(os.path.join(config_path,"config.json"), "w"), indent=4)

        model = LogicTreeNet(config.model_scale, 
                             in_ch=2, 
                             out_classes=config.out_classes, 
                             tau_gs=config.tau_gs, 
                             tau_noise=config.tau_noise, 
                             learn_tau_gs=config.lr_tau_gs == 0, 
                             input_size=config.target_size)
        
        criterion = nn.CrossEntropyLoss()

        # Load train/val/test split (80%/10%/10%)
        common_kwargs = dict(
            root_path=os.path.join(DATA_PATH, "N-Caltech101/Caltech101"),
            representation='binary',
            target_size=(config.target_size, config.target_size),
            train_split=config.train_size,
            val_split=0.1,
            seed=SEED
        )

        train_dataset = NCaltech101(
            transform=DATASET_TRANSFORM["N-Caltech101"]["train"],
            split='train',
            **common_kwargs
        )

        val_dataset = NCaltech101(
            transform=DATASET_TRANSFORM["N-Caltech101"]["val"],
            split='val',
            **common_kwargs
        )

        test_dataset = NCaltech101(
            transform=DATASET_TRANSFORM["N-Caltech101"]["val"],
            split='test',
            **common_kwargs
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                  generator=g, worker_init_fn=seed_worker, drop_last=True,
                                  pin_memory=True, persistent_workers=True, prefetch_factor=4)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=8,
                                generator=g, worker_init_fn=seed_worker, drop_last=True,
                                pin_memory=True, persistent_workers=True, prefetch_factor=4)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=8,
                                generator=g, worker_init_fn=seed_worker, drop_last=True,
                                pin_memory=True, persistent_workers=True, prefetch_factor=4)

        # separate parameters
        if config.lr_tau_gs == 0:
            # All parameters except tau
            other_params = [p for n, p in model.named_parameters() if n != "group_sum.tau"]
            # Tau parameter
            tau_params = [model.group_sum.tau]
            # Define optimizer with different learning rates
            params_list = [
                {"params": other_params, "lr": config.lr_model},
                {"params": tau_params, "lr": config.lr_tau_gs}
            ]
        else:
            # If tau is not learnable, use a single learning rate for all
            params_list = [{"params": model.parameters(), "lr": config.lr_model}]
        optimizer = torch.optim.AdamW(params_list, weight_decay=config.weight_decay)

        model.to(DEVICE)

        wandb.watch(model, criterion, log="all", log_freq=1000)

        train_loss = []
        val_loss = []
        val_accuracy = []
        best_accuracy = 0.0
        best_epoch = 0
        best_weights = {k:v.clone() for k,v in model.state_dict().items() if "weights" in k}
        pbar = tqdm(range(config.epochs), desc="training epochs")
        for epoch in pbar:
            epoch_train_loss, epoch_train_acc = train(model, train_loader, optimizer, criterion, DEVICE)
            train_loss.append(epoch_train_loss)

            epoch_val_loss, epoch_val_acc = evaluate(model, val_loader, criterion, DEVICE)
            val_loss.append(epoch_val_loss)
            val_accuracy.append(epoch_val_acc)

            if epoch_val_acc > best_accuracy:
                best_accuracy = epoch_val_acc
                best_epoch = epoch
                for k,v in model.state_dict().items():
                    if "weights" in k: best_weights[k] = v.clone()

            wandb.log({"epoch": epoch+1,
                       "train_loss": epoch_train_loss,
                       "val_loss": epoch_val_loss,
                       "train_accuracy": epoch_train_acc,
                       "val_accuracy": epoch_val_acc,
                       "tau_gs": model.group_sum.tau.detach().item(),
                       })

            pbar.set_postfix({"train loss (curr, prev)":
                            (f"{epoch_train_loss:.3f}", f"{train_loss[-2]:.3f}" if len(train_loss) > 1 else None),
                            "val loss":f"{epoch_val_loss:.3f}",
                            "train acc":f"{epoch_train_acc:.3f}",
                            "val acc":f"{epoch_val_acc:.3f}", })

        print("training finished")

        model.load_state_dict(best_weights, strict=False)
        print("restored best weights from epoch {}".format(best_epoch+1))

        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        wandb.log({"best_epoch": best_epoch+1,
                   "best_val_loss": val_loss[best_epoch],
                   "best_val_accuracy": val_accuracy[best_epoch],
                   "test_loss": test_loss,
                   "test_accuracy": test_acc})

        label, path_saved = save_model(model, ckpt_path)
        print("model saved to {}".format(path_saved))

        model_artifact = wandb.Artifact(label, type="model", metadata=dict(config))
        model_artifact.add_file(path_saved)
        wandb.save(path_saved, base_path=ckpt_path)
        wandb.log_artifact(model_artifact)
        print("wandb log completed\n------------------------")

if __name__ == "__main__":
    main()
