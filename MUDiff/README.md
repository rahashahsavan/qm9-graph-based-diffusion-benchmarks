## MUDiff — Sampling pretrained QM9 models (10k molecules)

This folder contains the MUDiff implementation used in: MUDiff: Unified Diffusion for Complete Molecule Generation. See the paper for the method and evaluation protocol: https://doi.org/10.48550/arXiv.2304.14621

### 1) Installation (inference-only)

Use Python 3.8–3.11 in a fresh virtual environment. The minimal requirements for sampling are pinned in `MUDiff/requirements_sampling.txt`.

CPU (quick start):

```bash
pip install -r MUDiff/requirements_sampling.txt
```

GPU (recommended): install a Torch build that matches your CUDA, then install the rest. Example for CUDA 11.7:

```bash
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r MUDiff/requirements_sampling.txt --no-deps
```

Notes
- PyTorch Geometric wheels must match your Torch/CUDA version. The pins in `requirements_sampling.txt` match Torch 1.13.1.
- RDKit is pulled from `rdkit-pypi`.

### 2) Pretrained QM9 checkpoints in `outputs/`

This repo already contains QM9-trained checkpoints under `MUDiff/outputs/` (subfolders such as `qm9_H_0407`, `qm9_noH_0407`, etc.). Each subfolder includes:
- `args.pickle`: training/eval configuration
- `generative_model.npy` and optionally `generative_model_ema.npy`: model weights (EMA preferred when present)

### 3) Sampling 10,000 molecules per model

We provide `MUDiff/sample_10k.py` which loads a checkpoint and writes XYZ-like `.txt` files to an `eval` subfolder. It uses the same dataloading, node-size distribution, and saving utilities as the paper’s evaluation code.

General command:

```bash
python -m MUDiff.sample_10k --model_path MUDiff/outputs/<MODEL_DIR> --n_samples 10000 --batch_size 128
```

Outputs will be saved under `MUDiff/outputs/<MODEL_DIR>/eval/molecules_10k/` as `molecule_XXX.txt` (and can be visualized with `qm9/visualizer.py`).

Example commands for the provided models:

```bash
# With hydrogens
python -m MUDiff.sample_10k --model_path MUDiff/outputs/qm9_H_0407 --n_samples 10000 --batch_size 128
python -m MUDiff.sample_10k --model_path MUDiff/outputs/qm9_H_0408 --n_samples 10000 --batch_size 128
python -m MUDiff.sample_10k --model_path MUDiff/outputs/qm9_H_0409 --n_samples 10000 --batch_size 128

# Without hydrogens
python -m MUDiff.sample_10k --model_path MUDiff/outputs/qm9_noH_0407 --n_samples 10000 --batch_size 128
python -m MUDiff.sample_10k --model_path MUDiff/outputs/qm9_noH_0408 --n_samples 10000 --batch_size 128
python -m MUDiff.sample_10k --model_path MUDiff/outputs/qm9_noH_0409 --n_samples 10000 --batch_size 128
```

Optional flags
- `--output_subdir`: change the output folder name (default: `eval/molecules_10k`).
- `--batch_size`: increase to speed up sampling if you have more GPU memory.

### 4) Visualizing generated molecules

After sampling, you can render PNGs or make GIFs with the provided visualizer:

```bash
python -c "from qm9 import visualizer as v; from configs.datasets_config import get_dataset_info; import pickle, os; p='MUDiff/outputs/qm9_H_0407'; d=get_dataset_info('qm9', False); v.visualize(os.path.join(p,'eval/molecules_10k'), d, max_num=100, spheres_3d=True)"
```

### 5) Consistency with the paper

The sampling path mirrors the paper’s setup:
- Loads `args.pickle` and the EMA checkpoint when available, consistent with diffusion sampling.
- Draws molecule sizes from the learned nodes distribution (`DistributionNodes`), matching the QM9 statistics used in training.
- Saves in the same XYZ-like format used by the repo’s evaluation utilities.

For further methodological details and evaluation protocol, refer to the paper: https://doi.org/10.48550/arXiv.2304.14621


