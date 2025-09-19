### DisCo (QM9) – Training with Lightweight Checkpoint/Resume

This repo contains a fork of DisCo for molecular graph generation, modified to add lightweight checkpointing and robust resume during training on QM9.

References:
- DisCo codebase: [GitHub: DisCo](https://github.com/pricexu/DisCo)
- Paper and hyperparameters: [OpenReview: Discrete-state Continuous-time Diffusion for Graph Generation](https://openreview.net/forum?id=YkSKZEhIYt&referrer=%5Bthe%20profile%20of%20Zhichen%20Zeng%5D(%2Fprofile%3Fid%3D~Zhichen_Zeng1))

### Requirements

- Python 3.9
- PyTorch 1.13.1
- torch-geometric 2.2.0 (+ related extensions)
- RDKit 2023.09.4
- Others: numpy 1.25.0, torchmetrics 1.2.1, scipy, pandas, tqdm, pyyaml, rich, networkx, pyemd

Install tips:
- PyTorch Geometric (GPU/CUDA 11.7):
```bash
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```
- conda install -c conda-forge rdkit=2023.09.4
- CPU-only:
```bash
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
```
- Common packages:
```bash
pip install numpy==1.25.0 torchmetrics==1.2.1 pyemd==1.0.0 scipy pandas tqdm pyyaml rich networkx rdkit-pypi==2023.9.4
```

Note: For graph benchmarks (sbm/planar/community), graph-tool and orca are needed (recommended on Linux/WSL2). Not required for QM9.

### Data Layout

- Datasets load under `data/`:
  - QM9: `data/qm9`
  - MOSES: `data/moses`
  - Guacamol: `data/guacamol`
- Dataset loaders in `loader/` handle download/preprocessing on first run.

### Train on QM9

Recommended (auto-resume + time-based autosave):
```bash
python train_mol.py --dataset qm9 --checkpoint_dir checkpoints/qm9 --resume --autosave_minutes 10
```

Key args:
- `--checkpoint_dir`: directory for checkpoints (default: `checkpoints/qm9`)
- `--resume`: resume from the latest checkpoint if present
- `--resume_from`: resume from a specific checkpoint file
- `--autosave_minutes`: time-based autosave interval (minutes)

Other useful args:
- `--batch_size` (default 128)
- `--epochs` (default 3000)
- `--device` (`cpu` or GPU index like `0`)
- Backbone: `--backbone {GT,MPNN}`
- Diffusion: `--diff_type`, `--min_time`, `--sampling_steps`, `--beta`, `--alpha`
- Model: `--lr`, `--wd`, `--dropout`, `--n_layers`, `--n_dim`

Run with YAML config:
```bash
python train_mol.py --dataset qm9 --config configs/qm9.yaml --resume
```
YAML keys under `checkpointing` supported: `checkpoint_dir`, `resume`, `resume_from`, `autosave_minutes`.

### How to train on QM9 and save only one final checkpoint:

Final-only save, with resume and 10-min autosave disabled (to keep it strictly final-only):
Note: --save_final_only disables per-epoch and time-based autosaves but still saves an interrupt checkpoint on Ctrl+C/SIGTERM.
```bash
python train_mol.py --dataset qm9 --backbone GT --diff_type marginal --alpha 0.8 --beta 2.0 --sampling_steps 100 --save_final_only --resume --autosave_minutes 0
```
If you want the more aggressive corruption per paper:
```bash
python train_mol.py --dataset qm9 --backbone GT --diff_type marginal --alpha 1.0 --beta 5.0 --sampling_steps 100 --save_final_only --resume --autosave_minutes 0
```
Where the final checkpoint is
`Saved as: checkpoints/qm9/ckpt_final.pt`

### Checkpoint/Resume Behavior

- End of each epoch: saves `ckpt_epoch{E}.pt`
- Time-based autosave: saves `ckpt_autosave_step{S}.pt` when `--autosave_minutes > 0`
- Safe interrupt (Ctrl+C/SIGTERM): saves `ckpt_interrupt_step{S}.pt`
- `--resume`: resumes from latest checkpoint in `checkpoint_dir`
- `--resume_from`: resumes from a specific file

This is lightweight by design: no per-step checkpointing to avoid heavy I/O for large models.

### Evaluation and Sampling

- During QM9 training, periodic sampling and basic metrics (validity, uniqueness, novelty) are printed.
- For MOSES/Guacamol, after training, you can generate samples and evaluate using their benchmark scripts.

### Performance Tips

- If VRAM is limited, reduce `--batch_size` and/or increase epochs.
- Lower `--autosave_minutes` (e.g., 5) if epochs are long and you want more frequent safety saves.

Troubleshooting:
- For PyG installation on Windows, ensure wheel versions match your PyTorch/CUDA. See `https://data.pyg.org/whl/`.


### How to sample later from the saved model
example 
```bash
python - << \"PY\"
import torch
from sampling import TauLeaping
from forward_diff import ForwardDiffusion
from dataset_info import get_dataset_info
from loader.load_qm9_data import QM9Dataset
from auxiliary_features import AuxFeatures, ExtraMolecularFeatures
from utils import to_dense, to_graph_list
from digress_models import GraphTransformer
import torch.nn.functional as F

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dataset_info = get_dataset_info('qm9')
train_set = QM9Dataset(root='data', split='train')
n_node_type = train_set[0].x.shape[-1]
n_edge_type = train_set[0].edge_attr.shape[-1]
E_marginal = torch.tensor(dataset_info.E_marginal).float().to(device)
X_marginal = torch.tensor(dataset_info.X_marginal).float().to(device)

extra = ExtraMolecularFeatures(dataset_info)
add_aux = AuxFeatures([True, True, False, True], dataset_info.max_n_nodes, extra)

# Build same model dims as training
example = train_set[0]
from utils import to_dense as _to_dense
X_t, E_t, y_t = add_aux(*_to_dense(example))
input_dims = {'X': X_t.shape[-1], 'E': E_t.shape[-1], 'y': y_t.shape[-1] + 1}
output_dims = {'X': n_node_type, 'E': n_edge_type, 'y': 0}
hidden_mlp_dims = {'X': 256, 'E': 256, 'y': 256}
hidden_dims = {'dx': 256, 'de': 256, 'dy': 256, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 256, 'dim_ffy': 256}

model = GraphTransformer(n_layers=10, input_dims=input_dims, hidden_mlp_dims=hidden_mlp_dims, hidden_dims=hidden_dims, output_dims=output_dims).to(device)

# Load checkpoint
ckpt = torch.load('checkpoints/qm9/ckpt_final.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Diffuser and sampler
diffuser = ForwardDiffusion(n_node_type, n_edge_type, forward_type='marginal', node_marginal=X_marginal, edge_marginal=E_marginal, device=device, time_exponential=2.0, time_base=0.8)
sampler = TauLeaping(n_node_type, n_edge_type, num_steps=100, min_t=0.01, add_auxiliary_feature=add_aux, device=device, BAR=False)

@torch.no_grad()
def generate(n_sample=32):
    n_node_dist = torch.distributions.categorical.Categorical(torch.tensor(dataset_info.n_node_distribution))
    n_node = n_node_dist.sample((n_sample,)).to(device)
    X, E, node_mask = sampler.sample(diffuser, model, n_node)
    return to_graph_list(X, E, n_node)

graphs = generate(100)
print('Generated graphs:', len(graphs))
PY
```
