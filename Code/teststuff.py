import os.path as osp
import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import SuiteSparseMatrixCollection, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from blossom import maxWeightMatching
from LineGraphConverter import ToLineGraph
import pickle
import numpy as np

np.random.seed(123)
torch.manual_seed(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.version.cuda)

data = GNNBenchmarkDataset('/Data', 'TSP', transform=NormalizeFeatures())
print(data)
print(data[0])

data = GNNBenchmarkDataset('/Data', 'MNIST', transform=NormalizeFeatures())
print(data)
print(data[0])