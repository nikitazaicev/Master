import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import SuiteSparseMatrixCollection, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from blossom import maxWeightMatching
from LineGraphConverter import ToLineGraph
from torch_geometric.loader import DataLoader
import pickle
torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.version.cuda)

demo = SuiteSparseMatrixCollection('/Data', 'Newman', 'netscience', transform=NormalizeFeatures())
print("dataset", len(demo))
dataset = TUDataset(root='/tmp/PROTEINS', name='PROTEINS')
print("proteins", len(dataset))

loader = DataLoader(dataset, batch_size=32, shuffle=True)
