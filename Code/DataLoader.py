import torch
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.data import Data
from blossom import maxWeightMatching
from LineGraphConverter import ToLineGraph
import pickle
import numpy as np

np.random.seed(123)
torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def LoadTestData():
        
    testdata = Data()
    testdata.edge_index = torch.tensor([[0,1,2,3,3],
                                        [1,2,3,0,4]], dtype=torch.long)
    testdata.edge_attr = torch.tensor([0.8,0.2,0.8,0.2,0.2])
    testdata.num_nodes = 5 
    testdata.num_edges = 5
    testdata.node_features = torch.ones([testdata.num_nodes])
    testdata.x = torch.ones([testdata.num_nodes,1])
    
    
    original = testdata.clone()
    line_graph = ToLineGraph(testdata, testdata.edge_attr)
    
    target = [1,0,3,2,-1]
    
    return [original], [line_graph], [target]

def LoadData(count=1000, datasetname='MNIST'):
    print("LOADING DATASETS")
    print("-------------------")
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
