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

def LoadData(count=1000, datasetname='MNIST'):
    print("LOADING DATASETS")
    try:
        file_name = 'data/target_data.pkl'
        with open(file_name, 'rb') as file:
            print(file_name, " loaded")
            target = pickle.load(file)
            
        file_name = 'data/converted_dataset.pkl'
        with open(file_name, 'rb') as file:
            print(file_name, " loaded")
            converted_dataset = pickle.load(file)
                
        return GNNBenchmarkDataset('/Data', datasetname, transform=NormalizeFeatures())[:count], converted_dataset[:count], target[:count]
    
    except Exception:
        print(file_name, " not found creating new datafile")
        print("Downloading initial dataset")
        dataset = GNNBenchmarkDataset('/Data', datasetname, transform=NormalizeFeatures())
        for g in dataset: g.num_edges = len(dataset[0].pos)
        target = []
        converted_dataset = []
        
        for dataitem in dataset[:count]:
            print("Blossom matching")
            blossominput = []
            for i in range(len(dataitem.edge_index[0])):
                blossominput.append((dataitem.edge_index[0][i].item(),
                                     dataitem.edge_index[1][i].item(),
                                     dataitem.edge_attr[i].item()))

            target.append(maxWeightMatching(blossominput))
            line_graph = ToLineGraph(dataitem, dataitem.edge_attr, verbose = False)
            converted_dataset.append(line_graph)
        
        with open(file_name, 'wb') as file:
            pickle.dump(target, file)
            print(f'Object successfully saved to "{file_name}"')
            
        file_name = 'data/converted_dataset.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(converted_dataset, file)
            print(f'Object successfully saved to "{file_name}"')
    
    original = GNNBenchmarkDataset('/Data', datasetname, transform=NormalizeFeatures())[:count]
    return original, converted_dataset, target
    
# original, converted_dataset, target = LoadData(5)
