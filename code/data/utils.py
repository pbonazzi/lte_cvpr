import torch
import os
import random
import numpy as np

def seed_worker(worker_id):
    """
    To fix randomness dataloeader (source: https://docs.pytorch.org/docs/stable/notes/randomness.html)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def create_output_dirs(wandb_run_name):
    """
    Creates structured directories for checkpoints, configs, and logs.
    """
    out_path = wandb_run_name
    ckpt_path = os.path.join(out_path, "checkpoints")
    config_path = os.path.join(out_path, "configs")
    log_path = os.path.join(out_path, "logs")

    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(config_path, exist_ok=True)
    os.makedirs(log_path,exist_ok=True)

    return ckpt_path, config_path, log_path
