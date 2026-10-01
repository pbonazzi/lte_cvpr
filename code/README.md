# DiffLogic: Differentiable Logic Neural Networks

A PyTorch implementation of differentiable logic layers with CUDA acceleration,
used to train LogicTreeNet on event-camera datasets.

## Features

- **Logic Layer** — differentiable logic operations over 16 binary ops per gate
- **Convolutional Logic Layer** — tree-structured convolutional variant
- **CUDA Acceleration** — custom kernels for training (a packed-bit inference
  path exists but is unfinished)
- **Event datasets** — N-MNIST, N-Cars, N-Caltech101, CIFAR10-DVS
- **Binary spike tensors** — event streams binned into binary input tensors

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- A CUDA toolkit whose major version matches the installed PyTorch CUDA build.
  `environment.yml` requests `pytorch-cuda=12.1`, and PyTorch's extension
  builder rejects a CUDA major-version mismatch when compiling.
- Conda

### Environment setup

1. Create the environment:

   ```bash
   conda env create -f environment.yml
   ```

2. Activate it:

   ```bash
   conda activate difflogic
   ```

3. Build the CUDA extensions:

   ```bash
   python setup.py build_ext --inplace
   ```

4. Configure environment variables:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` to set `DATA_PATH` and `OUTPUT_PATH`. The trainers load `.env`
   via `python-dotenv`; when a variable is unset they fall back to `./data` and
   `./outputs` respectively. Note `.env.example` ships `OUTPUT_PATH=./output`,
   which is not the code's fallback.

> **Known limitations of `environment.yml`.** It pins `python=3.8`, but all four
> preprocessing scripts in `data/datasets/` use `X | Y` annotations that are
> evaluated at runtime and need Python 3.10 or newer. It also lacks `aedat`,
> which `tonic` needs to open CIFAR10-DVS `.aedat4` files; without it every file
> fails with `No module named 'aedat'`. Until the file is updated, use Python
> 3.10+ and run `pip install aedat`. A combination known to work: Python 3.11,
> PyTorch 2.6 (CUDA 12.4), tonic 1.4.3, aedat 2.1.0.

## Usage

### Training

Entry points are run as modules from this directory. One suggested entry point
per dataset:

```bash
python -m scripts.train_nmnist_spike_tensor           --model_scale m
python -m scripts.train_ncars_spike_tensor            --model_scale m
python -m scripts.train_cifar10_spike_tensor          --model_scale m
python -m scripts.train_ncaltech101_topk_spike_tensor --model_scale m --top_k_classes 101
```

`scripts/` also holds alternative experiments — `train_ncaltech101.py`,
`train_ncars.py`, `train_cifar10_block_aux_proxy_anchor.py` — and the shared
helper `ncaltech101_topk_common.py`.

Flags are not uniform across trainers:

| Flag | N-MNIST | N-Cars | CIFAR10-DVS | N-Caltech101 top-k |
| --- | --- | --- | --- | --- |
| `--epochs` | 100 | 200 | 200 | 200 |
| `--model_scale` | `s` | `s` | `s` | `s` |
| `--num_time_bins` | 3 | 3 | 5 | 9 |
| `--binning_strategy` | `duration` | `duration` | `duration` | `duration` |
| `--run_name` | yes | yes | yes | yes |
| `--use_cache` | yes | yes | yes | yes |
| `--batch_size` | — | — | 32 | — |
| `--weights_init_mode` | — | — | `residual` | `residual` |
| `--top_k_classes` | — | — | — | 6 |

Cells show the default; `—` means the flag does not exist on that script.
`--model_scale` accepts `s,m,b,l,g`; `--binning_strategy` accepts
`duration,event_count`; `--weights_init_mode` accepts `residual,gaussian`.

`--top_k_classes` defaults to 6 on N-Caltech101 and trains on the six largest
categories only. Pass `--top_k_classes 101` for the full dataset.

### Datasets

Only CIFAR10-DVS downloads itself, into `DATA_PATH` on first use; it is a
10.4 GiB (11.2 GB) archive expanding to 10,000 `.aedat4` files across 10 class
folders.
N-MNIST, N-Cars and N-Caltech101 must be placed under `DATA_PATH` manually and
raise if their directories are missing:

```
DATA_PATH/N-MNIST/{Train,Test}/<digit>/*.bin
DATA_PATH/N-Cars/...
DATA_PATH/N-Caltech101/Caltech101/<class>/*.bin
DATA_PATH/CIFAR10-DVS/<class>/*.aedat4
```

Spike-tensor caches can be built ahead of time so training does not decode and
denoise events every epoch. Pass an explicit path — this script does not load
`.env`:

```bash
python data/datasets/cifar10_dvs_preprocess.py \
    --data_path /absolute/path/to/data --num_time_bins 5 --binning_strategy duration \
    --denoise_filter_time_us 50000
```

Then pass `--use_cache` to the trainer. The cache filename tag encodes the bin
count, binning strategy and denoise setting, and the trainer only uses files
whose tag matches its own settings. The preprocessor defaults to
`--denoise_filter_time_us 3000` but both CIFAR10-DVS trainers default to
`50000`, which is why the command above passes it explicitly. Tensors are cached
at sensor resolution and resized on load, so `target_size` is deliberately not
part of the tag.

### Using logic layers directly

```python
from difflogic import LogicLayer, ConvLogicLayer

logic_layer = LogicLayer(in_dim=512, out_dim=256)
conv_logic_layer = ConvLogicLayer(in_channels=3, out_channels=64)
```

## Configuration

Environment variables, read from `.env`:

- `DATA_PATH` — dataset directory (code fallback `./data`)
- `OUTPUT_PATH` — checkpoint and config output directory (code fallback `./outputs`)
- `WANDB_API_KEY` — optional, Weights & Biases API key
- `WANDB_PROJECT` — optional, W&B project; each script falls back to its own
  default project

Run name, group and tags can be set through wandb's own `WANDB_NAME`,
`WANDB_RUN_GROUP` and `WANDB_TAGS` variables.

## Project structure

```
code/
├── difflogic/
│   ├── __init__.py
│   ├── logic_layer.py            # LogicLayer, GroupSum
│   ├── conv_logic_layer.py       # tree-structured conv logic layer
│   ├── functional.py             # bin_op_s, connection sampling, GradFactor
│   ├── gumbel_noise.py           # Gumbel-softmax gate sampling
│   ├── packbitstensor.py         # packed-bit inference tensors
│   ├── compiled_model.py         # compiled/discretized model
│   └── cuda/
│       ├── difflogic.cpp
│       ├── difflogic_kernel.cu
│       ├── conv_difflogic.cpp
│       └── conv_difflogic_kernel.cu
├── models/
│   └── logictreenet.py           # LogicTreeNet
├── data/
│   ├── datasets/                 # nmnist, ncars, ncaltech101, cifar10_dvs (+ preprocessors)
│   ├── transforms/               # event and spike-tensor augmentations
│   └── utils.py
├── scripts/                      # trainers, alternative experiments, shared helpers
├── model_best.pth                # bundled checkpoint, provenance undocumented
├── setup.py                      # CUDA extension build
├── environment.yml
└── .env.example
```

## Behaviour worth knowing

- **Soft vs hard gates.** With `tau_noise = 0`, training uses `softmax` over the
  16 ops and evaluation uses `one_hot(argmax)`. `train_accuracy` and
  `val_accuracy` are therefore measured through two different forward functions
  of the same parameters, so their difference mixes generalization with the
  soft-to-hard discretization gap. Training accuracy is also accumulated while
  the weights are still changing across the epoch. With `tau_noise > 0` the
  training path uses Gumbel sampling instead.
- **GroupSum temperature.** `GroupSum` sums each class bucket and divides by a
  fixed `tau`. `320 * k` is not generally divisible by the class count, so
  buckets hold either `floor(320*k / n_classes)` or `ceil(...)` features. Bucket
  size is therefore roughly `320 * k / n_classes`, which means `tau` tuned at one
  class count is mis-scaled at another: at `k = 256` the bucket is ~13653
  features for 6 classes and ~811 for 101.
- **`tau` is fixed at `lr_tau_gs = 0`, by two different routes.** The CIFAR10-DVS
  and N-Caltech101 trainers set `learn_tau_gs = (lr_tau_gs == 0)`, making `tau`
  an `nn.Parameter` in its own optimizer group at learning rate 0. N-MNIST and
  N-Cars use the opposite convention, `learn_tau_gs = (lr_tau_gs > 0)`, so `tau`
  stays a plain scalar. Either way training never moves it.
- **Gate init.** `--weights_init_mode` defaults to `residual`. `gaussian` is
  reachable for ablations but trained poorly in every run recorded in this
  repository's history; see that commit for the measurements and their limits.

## Troubleshooting

### CUDA compilation issues

1. Ensure CUDA is installed and on your `PATH`.
2. Check that your PyTorch build matches your CUDA version:

   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```

3. Rebuild the extensions:

   ```bash
   python setup.py clean --all
   python setup.py build_ext --inplace
   ```

### CIFAR10-DVS download fails

If the archive arrives empty or fails its MD5 check, check the download URL.
For the archive this repository uses,
`figshare.com/ndownloader/files/38023437` was observed answering programs with
an AWS bot check (HTTP 202, empty body, header `x-amzn-waf-action: challenge`)
and no redirect, while `ndownloader.figshare.com/files/38023437` served the
file; the loader now uses the latter. Delete any partial
`CIFAR10DVS.zip` before retrying.
