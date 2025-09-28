from torch_geometric.datasets import QM9
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
import torch.nn.functional as F
from tqdm import tqdm
import torch
from torch import nn
import math
import wandb
import os

from models import DiffusionOrderingNetwork, DenoisingNetwork
from utils import NodeMasking
from grapharm import GraphARM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

# Choose configuration: with or without hydrogens
REMOVE_HYDROGEN = True  # Set to False to include hydrogens

# Load QM9 dataset
dataset = QM9(root='./data/QM9', transform=None, pre_transform=None)

# Process dataset to remove hydrogens if needed
if REMOVE_HYDROGEN:
    print("Removing hydrogens from QM9 dataset...")
    processed_data = []
    for data in tqdm(dataset):
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
    dataset = processed_data
else:
    print("Keeping all atoms including hydrogens")
    dataset = list(dataset)

# Get dataset statistics
all_x = torch.cat([data.x for data in dataset])
all_edge_attr = torch.cat([data.edge_attr for data in dataset])

num_node_types = len(all_x.unique())
num_edge_types = len(all_edge_attr.unique())
node_feature_dim = dataset[0].x.shape[1]
edge_feature_dim = dataset[0].edge_attr.shape[1]

print(f"Dataset statistics:")
print(f"  Number of molecules: {len(dataset)}")
print(f"  Node types: {num_node_types}")
print(f"  Edge types: {num_edge_types}")
print(f"  Node feature dim: {node_feature_dim}")
print(f"  Edge feature dim: {edge_feature_dim}")

# Initialize networks for QM9
diff_ord_net = DiffusionOrderingNetwork(
    node_feature_dim=node_feature_dim,
    num_node_types=num_node_types,
    num_edge_types=num_edge_types,
    num_layers=3,
    out_channels=1,
    hidden_dim=256,
    device=device
)

denoising_net = DenoisingNetwork(
    node_feature_dim=node_feature_dim,
    edge_feature_dim=edge_feature_dim,
    num_node_types=num_node_types,
    num_edge_types=num_edge_types,
    num_layers=5,  # L=5 as per paper
    hidden_dim=256,  # 256 as per paper
    K=20,  # K=20 as per paper
    device=device
)

# Initialize GraphARM
grapharm = GraphARM(
    dataset=dataset,
    denoising_network=denoising_net,
    diffusion_ordering_network=diff_ord_net,
    device=device
)

# Load pre-trained model if available
try:
    grapharm.load_model("qm9_denoising_network.pt", "qm9_diffusion_ordering_network.pt")
    print("Loaded pre-trained QM9 model")
except:
    print("No pre-trained model found. Please train the model first.")

# Training configuration
wandb.init(
    project="GraphARM_QM9",
    group="QM9_training",
    name="QM9_GraphARM",
    config={
        "dataset": "QM9",
        "n_epochs": 1000,
        "batch_size": 32,
        "lr_denoising": 1e-3,
        "lr_ordering": 5e-2,
    }
)

# Training loop
batch_size = 32
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:]

for epoch in range(1000):
    print(f"Epoch {epoch}")
    
    # Training
    for i in range(0, len(train_dataset), batch_size):
        train_batch = train_dataset[i:i+batch_size]
        val_batch = val_dataset[i:i+batch_size] if i < len(val_dataset) else val_dataset[:batch_size]
        
        denoising_loss, ordering_loss = grapharm.train_step(
            train_batch=train_batch,
            val_batch=val_batch,
            M=4
        )
        
        if i % 100 == 0:
            print(f"Batch {i}, Denoising Loss: {denoising_loss:.4f}, Ordering Loss: {ordering_loss:.4f}")
    
    # Save model every 100 epochs
    if epoch % 100 == 0:
        grapharm.save_model(f"qm9_denoising_network_epoch_{epoch}.pt", 
                          f"qm9_diffusion_ordering_network_epoch_{epoch}.pt")
        print(f"Model saved at epoch {epoch}")

print("Training completed!")
