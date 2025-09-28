#!/usr/bin/env python3
"""
Test script to verify QM9 dataset loading and processing
"""

from torch_geometric.datasets import QM9
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
import torch
from tqdm import tqdm

def test_qm9_loading():
    """Test QM9 dataset loading and hydrogen removal"""
    
    print("Testing QM9 dataset loading...")
    
    # Load QM9 dataset
    dataset = QM9(root='./data/QM9', transform=None, pre_transform=None)
    print(f"Original dataset size: {len(dataset)}")
    
    # Check first molecule
    first_mol = dataset[0]
    print(f"First molecule:")
    print(f"  Node features shape: {first_mol.x.shape}")
    print(f"  Edge index shape: {first_mol.edge_index.shape}")
    print(f"  Edge attributes shape: {first_mol.edge_attr.shape}")
    print(f"  Node types: {first_mol.x.unique()}")
    
    # Test hydrogen removal
    print("\nTesting hydrogen removal...")
    processed_data = []
    
    for i, data in enumerate(tqdm(dataset[:100])):  # Test with first 100 molecules
        # Get atom types (assuming one-hot encoding)
        if data.x.shape[1] > 1:  # One-hot encoded
            atom_types = torch.argmax(data.x, dim=1)
        else:  # Already encoded
            atom_types = data.x.squeeze()
        
        # Keep only heavy atoms (C, N, O, F) - types 1, 2, 3, 4
        to_keep = atom_types > 0  # Remove H (type 0)
        
        if to_keep.sum() > 0:  # Only keep molecules with heavy atoms
            # Update edge_index and edge_attr
            edge_index, edge_attr = subgraph(to_keep, data.edge_index, data.edge_attr, 
                                           relabel_nodes=True, num_nodes=len(to_keep))
            
            # Update node features
            x = data.x[to_keep]
            if x.shape[1] > 1:  # One-hot encoded
                x = x[:, 1:]  # Remove H column
            
            # Create new data object
            new_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            processed_data.append(new_data)
    
    print(f"Processed {len(processed_data)} molecules (removed hydrogens)")
    
    if processed_data:
        # Check processed molecule
        first_processed = processed_data[0]
        print(f"First processed molecule:")
        print(f"  Node features shape: {first_processed.x.shape}")
        print(f"  Edge index shape: {first_processed.edge_index.shape}")
        print(f"  Edge attributes shape: {first_processed.edge_attr.shape}")
        print(f"  Node types: {first_processed.x.unique()}")
        
        # Get dataset statistics
        all_x = torch.cat([data.x for data in processed_data])
        all_edge_attr = torch.cat([data.edge_attr for data in processed_data])
        
        num_node_types = len(all_x.unique())
        num_edge_types = len(all_edge_attr.unique())
        node_feature_dim = processed_data[0].x.shape[1]
        edge_feature_dim = processed_data[0].edge_attr.shape[1]
        
        print(f"\nDataset statistics:")
        print(f"  Number of molecules: {len(processed_data)}")
        print(f"  Node types: {num_node_types}")
        print(f"  Edge types: {num_edge_types}")
        print(f"  Node feature dim: {node_feature_dim}")
        print(f"  Edge feature dim: {edge_feature_dim}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_qm9_loading()
