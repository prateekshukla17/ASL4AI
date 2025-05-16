import os
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np

class ASLDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data_dir = os.path.join(root, "graphs")  # folder with .npz files or .pt graphs
        self.graph_files = [f for f in os.listdir(self.data_dir) if f.endswith(".npz") or f.endswith(".pt")]
def __len__(self):
    return len(self.graph_files)

def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.graph_files[idx])
        
        if path.endswith(".npz"):
            
            graph = np.load(path)
            x = torch.tensor(graph["x"], dtype=torch.float)               # [num_nodes, num_features]
            edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)  # [2, num_edges]
            y = torch.tensor(graph["y"], dtype=torch.long)               # scalar or [num_nodes]
        elif path.endswith(".pt"):
            return torch.load(path) 
        else:
            raise ValueError("Unsupported graph file format")

        return Data(x=x, edge_index=edge_index, y=y)

# Utility function to get loaders
def get_dataloaders(root, batch_size=1, shuffle=True):
    dataset = ASLDataset(root=root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)