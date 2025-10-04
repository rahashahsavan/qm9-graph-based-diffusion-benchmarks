#!/usr/bin/env python3
"""
Simple molecule generation script for DisCo QM9 model
Only generates molecules and saves them - no evaluation
Based on the original DisCo paper parameters
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse

# DisCo imports
from sampling import TauLeaping
from forward_diff import ForwardDiffusion
from dataset_info import get_dataset_info
from loader.load_qm9_data import QM9Dataset
from auxiliary_features import AuxFeatures, ExtraMolecularFeatures
from utils import to_dense, to_graph_list
from digress_models import GraphTransformer
from rdkit_functions import graph2smile

def load_trained_model(checkpoint_path, device='cuda:0'):
    """Load the trained DisCo model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # Setup dataset info and auxiliary features
    dataset_info = get_dataset_info('qm9')
    train_set = QM9Dataset(root='data', split='train')
    n_node_type = train_set[0].x.shape[-1]
    n_edge_type = train_set[0].edge_attr.shape[-1]
    
    E_marginal = torch.tensor(dataset_info.E_marginal).float().to(device)
    X_marginal = torch.tensor(dataset_info.X_marginal).float().to(device)
    
    extra = ExtraMolecularFeatures(dataset_info)
    add_aux = AuxFeatures([True, True, False, True], dataset_info.max_n_nodes, extra)
    
    # Build model with same dimensions as training
    example = train_set[0]
    X_t, E_t, y_t = add_aux(*to_dense(example))
    input_dims = {'X': X_t.shape[-1], 'E': E_t.shape[-1], 'y': y_t.shape[-1] + 1}
    output_dims = {'X': n_node_type, 'E': n_edge_type, 'y': 0}
    hidden_mlp_dims = {'X': 256, 'E': 256, 'y': 256}
    hidden_dims = {'dx': 256, 'de': 256, 'dy': 256, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 256, 'dim_ffy': 256}
    
    model = GraphTransformer(n_layers=10, input_dims=input_dims, hidden_mlp_dims=hidden_mlp_dims, 
                           hidden_dims=hidden_dims, output_dims=output_dims).to(device)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {ckpt.get('epoch', 'unknown')}, step {ckpt.get('global_step', 'unknown')}")
    
    # Setup diffuser and sampler with paper parameters
    # From your training: alpha=1.0, beta=5.0, sampling_steps=100
    diffuser = ForwardDiffusion(n_node_type, n_edge_type, forward_type='marginal', 
                               node_marginal=X_marginal, edge_marginal=E_marginal, 
                               device=device, time_exponential=5.0, time_base=1.0)
    
    sampler = TauLeaping(n_node_type, n_edge_type, num_steps=100, min_t=0.01, 
                        add_auxiliary_feature=add_aux, device=device, BAR=False)
    
    return model, diffuser, sampler, dataset_info

@torch.no_grad()
def generate_molecules(model, diffuser, sampler, dataset_info, n_samples=10000, batch_size=32, device='cuda:0'):
    """
    Generate molecules using the trained DisCo model
    
    Args:
        model: Trained DisCo model
        diffuser: Forward diffusion process
        sampler: Tau-leaping sampler
        dataset_info: QM9 dataset information
        n_samples: Total number of molecules to generate
        batch_size: Batch size for generation (32 is paper default, adjust for memory)
        device: Device to use
    
    Returns:
        all_graphs: List of generated graphs [(atom_types, edge_types), ...]
        all_smiles: List of valid SMILES strings
        generation_time: Total generation time in seconds
    """
    print(f"Generating {n_samples} molecules using DisCo model")
    print(f"Batch size: {batch_size}")
    print(f"Sampling steps: 100 (tau-leaping)")
    print(f"Device: {device}")
    
    model.eval()
    all_graphs = []
    all_smiles = []
    
    # Use QM9 node size distribution from dataset
    n_node_dist = torch.distributions.categorical.Categorical(torch.tensor(dataset_info.n_node_distribution))
    
    start_time = time.time()
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(n_batches), desc="Generating batches"):
        current_batch_size = min(batch_size, n_samples - i * batch_size)
        
        # Sample number of nodes for each molecule from QM9 distribution
        n_node = n_node_dist.sample((current_batch_size,)).to(device)
        
        # Generate molecules using tau-leaping sampler
        X, E, node_mask = sampler.sample(diffuser, model, n_node)
        
        # Convert to graph list format
        graphs = to_graph_list(X, E, n_node)
        
        # Convert to SMILES and store
        for graph in graphs:
            all_graphs.append(graph)
            smile = graph2smile(graph, dataset_info.atom_decoder)
            if smile is not None:
                all_smiles.append(smile)
    
    generation_time = time.time() - start_time
    
    print(f"\nGeneration completed!")
    print(f"Total molecules generated: {len(all_graphs)}")
    print(f"Valid SMILES: {len(all_smiles)} ({len(all_smiles)/len(all_graphs)*100:.1f}%)")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Time per molecule: {generation_time/n_samples*1000:.2f} ms")
    
    return all_graphs, all_smiles, generation_time

def save_molecules(graphs, smiles, output_dir='generated_molecules', prefix='disco_qm9'):
    """Save generated molecules in multiple formats for later evaluation"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save graphs as pickle (for later evaluation)
    graphs_file = os.path.join(output_dir, f'{prefix}_graphs.pkl')
    with open(graphs_file, 'wb') as f:
        pickle.dump(graphs, f)
    print(f"Graphs saved to: {graphs_file}")
    
    # Save SMILES as text file
    smiles_file = os.path.join(output_dir, f'{prefix}_smiles.txt')
    with open(smiles_file, 'w') as f:
        for smile in smiles:
            f.write(smile + '\n')
    print(f"SMILES saved to: {smiles_file}")
    
    # Save SMILES as pickle (for later evaluation)
    smiles_pkl = os.path.join(output_dir, f'{prefix}_smiles.pkl')
    with open(smiles_pkl, 'wb') as f:
        pickle.dump(smiles, f)
    print(f"SMILES pickle saved to: {smiles_pkl}")
    
    # Save generation info
    info_file = os.path.join(output_dir, f'{prefix}_info.txt')
    with open(info_file, 'w') as f:
        f.write(f"DisCo QM9 Molecule Generation\n")
        f.write(f"Generated: {len(graphs)} molecules\n")
        f.write(f"Valid SMILES: {len(smiles)} ({len(smiles)/len(graphs)*100:.1f}%)\n")
        f.write(f"Model: DisCo with Graph Transformer backbone\n")
        f.write(f"Dataset: QM9\n")
        f.write(f"Sampling: Tau-leaping with 100 steps\n")
    print(f"Generation info saved to: {info_file}")
    
    return graphs_file, smiles_file

def main():
    parser = argparse.ArgumentParser(description='Generate molecules using trained DisCo model')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--n_samples', type=int, default=10000, 
                       help='Number of molecules to generate (default: 10000)')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for generation (default: 32, adjust for GPU memory)')
    parser.add_argument('--device', type=str, default='cuda:0', 
                       help='Device to use (default: cuda:0)')
    parser.add_argument('--output_dir', type=str, default='generated_molecules', 
                       help='Output directory (default: generated_molecules)')
    parser.add_argument('--prefix', type=str, default='disco_qm9', 
                       help='Prefix for output files (default: disco_qm9)')
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    print("=" * 60)
    print("DisCo QM9 Molecule Generation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Target samples: {args.n_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load model
    print("Loading trained DisCo model...")
    model, diffuser, sampler, dataset_info = load_trained_model(args.checkpoint, args.device)
    
    # Generate molecules
    print("Starting molecule generation...")
    graphs, smiles, gen_time = generate_molecules(
        model, diffuser, sampler, dataset_info, 
        n_samples=args.n_samples, batch_size=args.batch_size, device=args.device
    )
    
    # Save results
    print("Saving generated molecules...")
    graphs_file, smiles_file = save_molecules(graphs, smiles, args.output_dir, args.prefix)
    
    print("\n" + "=" * 60)
    print("Generation completed successfully!")
    print("=" * 60)
    print(f"Files saved in: {args.output_dir}/")
    print(f"- Graphs: {os.path.basename(graphs_file)}")
    print(f"- SMILES: {os.path.basename(smiles_file)}")
    print("\nThese files can be used for later evaluation with your preferred metrics.")

if __name__ == '__main__':
    main()


