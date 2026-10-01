# lte_cvpr

Logic Tree Networks (LogicTreeNet) for event-based vision. Differentiable logic
gate networks trained on event camera datasets, with custom CUDA kernels and a
binary spike-tensor input representation.

All code lives in [`code/`](code/). See [`code/README.md`](code/README.md) for
setup and usage.

## Contents

Paths are relative to the repository root.

| Path | What it is |
| --- | --- |
| `code/difflogic/` | Differentiable logic layers and their CUDA kernels |
| `code/models/logictreenet.py` | LogicTreeNet: 4 conv logic blocks + 3 dense logic layers + GroupSum |
| `code/data/datasets/` | N-MNIST, N-Cars, N-Caltech101, CIFAR10-DVS loaders and preprocessors |
| `code/data/transforms/` | Event and spike-tensor augmentations |
| `code/scripts/` | Training entry points, alternative experiments, shared N-Caltech101 helpers |
| `code/setup.py` | CUDA extension build |
| `code/environment.yml`, `code/.env.example` | Environment and configuration templates |
| `code/model_best.pth` | Bundled checkpoint: ResNet-34-shaped classifier plus a learned quantization MLP, with `class_names` for a 10-category N-Caltech101 subset. Provenance and evaluation procedure are undocumented, and no code in this repository loads it. |

## Datasets

| Dataset | Classes | Loader (under `code/`) | Obtaining it |
| --- | --- | --- | --- |
| N-MNIST | 10 | `data/datasets/nmnist.py` | place manually at `DATA_PATH/N-MNIST/{Train,Test}/<digit>/*.bin` |
| N-Cars | 2 | `data/datasets/ncars.py` | place manually under `DATA_PATH/N-Cars/` |
| N-Caltech101 | 101 | `data/datasets/ncaltech101.py` | place manually at `DATA_PATH/N-Caltech101/Caltech101/<class>/*.bin` |
| CIFAR10-DVS | 10 | `data/datasets/cifar10_dvs.py` | downloads automatically on first use (10.4 GiB) |

Only CIFAR10-DVS has a downloader; the other three raise if their directories
are missing. Set `DATA_PATH` to the directory holding the dataset folders.

## Quick start

```bash
cd code
conda env create -f environment.yml
conda activate difflogic
python setup.py build_ext --inplace
cp .env.example .env          # then edit DATA_PATH / OUTPUT_PATH
```

`environment.yml` currently pins Python 3.8 and lacks `aedat`, which the
preprocessing scripts and CIFAR10-DVS need. See the known limitations in
[`code/README.md`](code/README.md#environment-setup) for a working combination.

Train:

```bash
python -m scripts.train_nmnist_spike_tensor           --model_scale m
python -m scripts.train_ncars_spike_tensor            --model_scale m
python -m scripts.train_cifar10_spike_tensor          --model_scale m
python -m scripts.train_ncaltech101_topk_spike_tensor --model_scale m --top_k_classes 101
```

`--top_k_classes` defaults to **6**, which trains on the six largest
N-Caltech101 categories only. Pass `--top_k_classes 101` for the full dataset;
numbers from the two settings are not comparable.

## Model scales

`--model_scale` selects the base width `k`, giving conv widths `k, 4k, 16k, 32k`
and dense widths `1280k, 640k, 320k`:

| Scale | `k` |
| --- | --- |
| `s` | 32 |
| `m` | 256 |
| `b` | 512 |
| `l` | 1024 |
| `g` | 2560 |

Scale `m` has **668,416 independently parameterized gate choices**
(`7·(1+4+16+32)·k` in the conv trees plus `2240·k` in the dense layers). Since a
gate resolves to one of 16 binary ops, those op identities alone would occupy
0.334 MB at 4 bits each. That is a deployed-representation figure and excludes
wiring, `tau` and everything else: during training each gate carries 16 float32
logits, so the gate parameters actually occupy 668,416 × 16 × 4 B ≈ 42.8 MB, and
checkpoints store that, not the packed form.

## Notes

- With `tau_noise=0` (the spike-tensor trainers' setting) training uses **soft**
  gates (softmax over the 16 ops) and evaluation uses **hard** gates (`argmax`
  one-hot). Logged `train_accuracy` and `val_accuracy` therefore come from two
  different forward functions of the same parameters, and the gap between them
  is not purely generalization. With Gumbel noise enabled (`tau_noise > 0`) the
  training path differs again.
- `GroupSum` sums each class bucket and divides by a fixed `tau`. `320 · k` is
  not generally divisible by the class count, so buckets hold either
  `floor(320·k / n_classes)` or `ceil(...)` features. A `tau` tuned at one class
  count is therefore mis-scaled at another: at `k=256` the bucket goes from
  ~13653 features at 6 classes to ~811 at 101.
- Gate init defaults to `residual`. `gaussian` is available for ablations, but
  trained poorly in every run recorded here — see the `weights_init_mode`
  commit for the measurements and their limits.
