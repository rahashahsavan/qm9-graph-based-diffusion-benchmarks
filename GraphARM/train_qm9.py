from torch_geometric.datasets import QM9
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

# Load QM9 dataset
# Choose configuration: with or without hydrogens
REMOVE_HYDROGEN = True  # Set to False to include hydrogens

dataset = QM9(root='./data/QM9', transform=None, pre_transform=None, remove_h=REMOVE_HYDROGEN)

# Initialize networks for QM9
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
