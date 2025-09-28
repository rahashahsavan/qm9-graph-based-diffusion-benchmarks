import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from models import DiffusionOrderingNetwork, DenoisingNetwork
from utils import NodeMasking
from grapharm import GraphARM
import pickle
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import os

def generate_molecules(grapharm, num_molecules=10000, max_nodes=9, device='cuda', include_hydrogen=False):
    """
    Generate molecules using the trained GraphARM model
    """
    generated_molecules = []
    valid_molecules = 0
    
    print(f"Generating {num_molecules} molecules...")
    
    for i in tqdm(range(num_molecules)):
        try:
            # Start with a fully masked graph
            if include_hydrogen:
                n_nodes = np.random.randint(1, 29 + 1)  # Up to 29 atoms with H
            else:
                n_nodes = np.random.randint(1, max_nodes + 1)  # Up to 9 heavy atoms
            graph = grapharm.masker.generate_fully_masked(n_nodes)
            
            # Generate nodes one by one
            for node_idx in range(n_nodes):
                # Predict new node
                node_type, edge_types = grapharm.predict_new_node(
                    graph, 
                    sampling_method="sample", 
                    preprocess=True
                )
                
                # Add the predicted node
                graph = grapharm.masker.demask_node(
                    graph, 
                    node_idx, 
                    node_type, 
                    edge_types
                )
            
            # Convert to molecule if possible
            molecule_data = convert_graph_to_molecule(graph)
            if molecule_data is not None:
                generated_molecules.append(molecule_data)
                valid_molecules += 1
                
        except Exception as e:
            print(f"Error generating molecule {i}: {e}")
            continue
    
    print(f"Generated {valid_molecules} valid molecules out of {num_molecules}")
    return generated_molecules

def convert_graph_to_molecule(graph):
    """
    Convert PyTorch Geometric graph to RDKit molecule
    """
    try:
        # Extract node types (atom types)
        node_types = graph.x.squeeze().cpu().numpy()
        
        # Extract edge information
        edge_index = graph.edge_index.cpu().numpy()
        edge_attr = graph.edge_attr.cpu().numpy()
        
        # Create molecule
        mol = Chem.RWMol()
        
        # Add atoms
        atom_map = {}
        for i, atom_type in enumerate(node_types):
            if atom_type < 5:  # Valid atom types for QM9
                atom_symbols = ['C', 'N', 'O', 'F', 'H']
                atom = Chem.Atom(atom_symbols[int(atom_type)])
                atom_idx = mol.AddAtom(atom)
                atom_map[i] = atom_idx
        
        # Add bonds
        for edge_idx in range(edge_index.shape[1]):
            i, j = edge_index[:, edge_idx]
            bond_type = edge_attr[edge_idx]
            
            if i in atom_map and j in atom_map and bond_type > 0:
                bond_order = int(bond_type)
                if bond_order > 0:
                    mol.AddBond(atom_map[i], atom_map[j], Chem.BondType(bond_order))
        
        # Convert to molecule
        mol = mol.GetMol()
        if mol is not None:
            # Sanitize molecule
            Chem.SanitizeMol(mol)
            return mol
            
    except Exception as e:
        print(f"Error converting graph to molecule: {e}")
        return None
    
    return None

def save_molecules(molecules, filename="generated_molecules.smi"):
    """
    Save generated molecules to SMILES file
    """
    with open(filename, 'w') as f:
        for mol in molecules:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                f.write(f"{smiles}\n")
    
    print(f"Saved {len(molecules)} molecules to {filename}")

def analyze_molecules(molecules):
    """
    Analyze generated molecules
    """
    if not molecules:
        print("No molecules to analyze")
        return
    
    print(f"\n=== Molecule Analysis ===")
    print(f"Total molecules: {len(molecules)}")
    
    # Calculate molecular properties
    mol_weights = []
    num_atoms = []
    num_rings = []
    
    for mol in molecules:
        if mol is not None:
            mol_weights.append(rdMolDescriptors.CalcExactMolWt(mol))
            num_atoms.append(mol.GetNumAtoms())
            num_rings.append(rdMolDescriptors.CalcNumRings(mol))
    
    if mol_weights:
        print(f"Average molecular weight: {np.mean(mol_weights):.2f}")
        print(f"Average number of atoms: {np.mean(num_atoms):.2f}")
        print(f"Average number of rings: {np.mean(num_rings):.2f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load QM9 dataset for initialization
    # Choose configuration: with or without hydrogens
    REMOVE_HYDROGEN = True  # Set to False to include hydrogens
    
    dataset = QM9(root='./data/QM9', transform=None, pre_transform=None, remove_h=REMOVE_HYDROGEN)
    
    # Initialize networks
    diff_ord_net = DiffusionOrderingNetwork(
        node_feature_dim=1,
        num_node_types=dataset.x.unique().shape[0],
        num_edge_types=dataset.edge_attr.unique().shape[0],
        num_layers=3,
        out_channels=1,
        hidden_dim=256,
        device=device
    )
    
    denoising_net = DenoisingNetwork(
        node_feature_dim=dataset.num_features,
        edge_feature_dim=dataset.num_edge_features,
        num_node_types=dataset.x.unique().shape[0],
        num_edge_types=dataset.edge_attr.unique().shape[0],
        num_layers=5,
        hidden_dim=256,
        K=20,
        device=device
    )
    
    # Initialize GraphARM
    grapharm = GraphARM(
        dataset=dataset,
        denoising_network=denoising_net,
        diffusion_ordering_network=diff_ord_net,
        device=device
    )
    
    # Load trained model
    try:
        grapharm.load_model("qm9_denoising_network.pt", "qm9_diffusion_ordering_network.pt")
        print("Loaded trained QM9 model")
    except:
        print("No trained model found. Please train the model first using train_qm9.py")
        return
    
    # Generate molecules
    molecules = generate_molecules(grapharm, num_molecules=10000, max_nodes=9, device=device, include_hydrogen=not REMOVE_HYDROGEN)
    
    # Save molecules
    save_molecules(molecules, "qm9_generated_10000.smi")
    
    # Analyze molecules
    analyze_molecules(molecules)
    
    print("Molecule generation completed!")

if __name__ == "__main__":
    main()
