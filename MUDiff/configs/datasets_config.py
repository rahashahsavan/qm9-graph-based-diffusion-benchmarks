"""
Dataset configuration for MUDiff QM9 models.
Based on the paper: MUDiff: Unified Diffusion for Complete Molecule Generation
https://doi.org/10.48550/arXiv.2304.14621
"""

import torch
import numpy as np


# QM9 dataset with hydrogens (remove_h=False)
qm9_with_h = {
    'name': 'qm9',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'n_nodes': {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0, 24: 0.0, 25: 0.0, 26: 0.0, 27: 0.0, 28: 0.0, 29: 1.0},
    'max_n_nodes': 29,
    'edge_types': [0, 1, 2, 3, 4],  # No bond, single, double, triple, aromatic
    'atom_weights': {0: 1, 1: 12, 2: 14, 3: 16, 4: 19},
    'max_weight': 150,
    'max_in_deg': 4,
    'max_out_deg': 4,
    'max_num_edges': 4,
    'max_spatial': 4,
    'max_edge_dist': 4,
    'colors_dic': ['#FFFFFF', '#909090', '#3050F8', '#FF0D0D', '#90E050'],
    'radius_dic': [0.32, 0.76, 0.65, 0.60, 0.56],
    'remove_h': False
}

# QM9 dataset without hydrogens (remove_h=True)
qm9_without_h = {
    'name': 'qm9',
    'atom_encoder': {'C': 0, 'N': 1, 'O': 2, 'F': 3},
    'atom_decoder': ['C', 'N', 'O', 'F'],
    'n_nodes': {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 1.0},
    'max_n_nodes': 9,
    'edge_types': [0, 1, 2, 3, 4],  # No bond, single, double, triple, aromatic
    'atom_weights': {0: 12, 1: 14, 2: 16, 3: 19},
    'max_weight': 150,
    'max_in_deg': 4,
    'max_out_deg': 4,
    'max_num_edges': 4,
    'max_spatial': 4,
    'max_edge_dist': 4,
    'colors_dic': ['#909090', '#3050F8', '#FF0D0D', '#90E050'],
    'radius_dic': [0.76, 0.65, 0.60, 0.56],
    'remove_h': True
}

# GEOM dataset (for reference)
geom_with_h = {
    'name': 'geom',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Br': 8, 'I': 9, 'B': 10, 'Si': 11, 'Se': 12, 'Te': 13, 'As': 14, 'Al': 15},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'Te', 'As', 'Al'],
    'n_nodes': {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0, 24: 0.0, 25: 0.0, 26: 0.0, 27: 0.0, 28: 0.0, 29: 0.0, 30: 0.0, 31: 0.0, 32: 0.0, 33: 0.0, 34: 0.0, 35: 0.0, 36: 0.0, 37: 0.0, 38: 0.0, 39: 0.0, 40: 0.0, 41: 0.0, 42: 0.0, 43: 0.0, 44: 1.0},
    'max_n_nodes': 44,
    'edge_types': [0, 1, 2, 3, 4],
    'atom_weights': {0: 1, 1: 12, 2: 14, 3: 16, 4: 19, 5: 31, 6: 32, 7: 35, 8: 80, 9: 127, 10: 11, 11: 28, 12: 79, 13: 128, 14: 75, 15: 27},
    'max_weight': 200,
    'max_in_deg': 4,
    'max_out_deg': 4,
    'max_num_edges': 4,
    'max_spatial': 4,
    'max_edge_dist': 4,
    'colors_dic': ['#FFFFFF', '#909090', '#3050F8', '#FF0D0D', '#90E050', '#FF8000', '#FFFF30', '#1FF01F', '#8A00FF', '#A0A0A0', '#FF1493', '#00FFFF', '#FFD700', '#FF69B4', '#FF4500', '#C0C0C0'],
    'radius_dic': [0.32, 0.76, 0.65, 0.60, 0.56, 1.10, 1.05, 1.00, 1.20, 1.40, 0.85, 1.10, 1.20, 1.40, 1.20, 1.25],
    'remove_h': False
}


def get_dataset_info(dataset_name, remove_h=False):
    """
    Get dataset information for MUDiff models.
    
    Args:
        dataset_name (str): Name of the dataset ('qm9', 'geom', etc.)
        remove_h (bool): Whether to remove hydrogens from the dataset
    
    Returns:
        dict: Dataset configuration dictionary
    """
    if dataset_name == 'qm9':
        if remove_h:
            return qm9_without_h
        else:
            return qm9_with_h
    elif dataset_name == 'geom':
        return geom_with_h
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
