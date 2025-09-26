#!/usr/bin/env python
# Rdkit import should be first, do not move it
try:
    from rdkit import Chem  # noqa: F401
except ModuleNotFoundError:
    pass

import argparse
import os
import pickle
from os.path import join

import torch

from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.models import get_model
from qm9.sampling import sample
from qm9 import visualizer as vis


def load_trained_model(model_path: str, device: torch.device):
    with open(join(model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    # Backward compatibility for missing args
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    args.cuda = torch.cuda.is_available()
    args.device = device

    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    dataloaders, _ = dataset.retrieve_dataloaders(args)

    model, nodes_dist = get_model(args, dataset_info)
    model.to(device)

    # Prefer EMA weights if present, per common practice in diffusion models
    weight_filename = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    weight_path = join(model_path, weight_filename)
    if not os.path.exists(weight_path):
        # Fallback to non-EMA if EMA missing
        weight_path = join(model_path, 'generative_model.npy')

    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)

    return args, model, nodes_dist, dataset_info


def main():
    parser = argparse.ArgumentParser(description='Sample N molecules from a trained MUDiff model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to folder containing args.pickle and weights')
    parser.add_argument('--n_samples', type=int, default=10000, help='Total number of molecules to generate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for sampling')
    parser.add_argument('--output_subdir', type=str, default='eval/molecules_10k', help='Relative output directory under model_path')
    args_cli = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args, model, nodes_dist, dataset_info = load_trained_model(args_cli.model_path, device)
    model.eval()

    save_dir = join(args_cli.model_path, args_cli.output_subdir)
    os.makedirs(save_dir, exist_ok=True)

    num_saved = 0
    while num_saved < args_cli.n_samples:
        current_bs = min(args_cli.batch_size, args_cli.n_samples - num_saved)
        nodesxsample = nodes_dist.sample(current_bs)

        with torch.no_grad():
            one_hot, charges, x, edge, node_mask = sample(
                args=args,
                device=device,
                generative_model=model,
                dataset_info=dataset_info,
                nodesxsample=nodesxsample,
            )

        vis.save_xyz_file(
            save_dir + '/',
            one_hot=one_hot,
            charges=charges,
            positions=x,
            adj=edge,
            dataset_info=dataset_info,
            id_from=num_saved,
            name='molecule',
            node_mask=node_mask,
        )

        num_saved += current_bs
        print(f'Saved {num_saved}/{args_cli.n_samples}')

    print('Done.')


if __name__ == '__main__':
    main()


