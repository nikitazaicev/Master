import torch
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import GNNBenchmarkDataset, KarateClub, TUDataset
from torch_geometric.data import Data
from blossom import maxWeightMatching
from LineGraphConverter import ToLineGraph
import pickle
import numpy as np
import copy

np.random.seed(123)
torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def RemoveDoubleEdges(graph):
    unique = set()
    idx = dict()
    deleted = 0
    total_edges = len(graph.edge_index[0])
    for i in range(total_edges):
        from_node = graph.edge_index[0][i].item()
        to_node = graph.edge_index[1][i].item()
        if (from_node,to_node) in unique or (to_node, from_node) in unique:
            deleted += 1
            continue
        unique.add((from_node,to_node))
        idx[(from_node,to_node)] = i
    new_edges = torch.zeros([2,len(unique)],dtype=torch.int)
    new_weights = torch.zeros([len(unique)],dtype=torch.float)
    new_atrs = torch.zeros([len(unique)],dtype=torch.float)
    unique  = sorted(unique)
    for i, u  in enumerate(unique):
        
        new_edges[0][i] = torch.tensor(u[0])
        new_edges[1][i] = torch.tensor(u[1])   
        prev_id = idx[(u[0],u[1])]
        
        try:
            new_weights[prev_id] = graph.edge_weight[prev_id]
        except Exception:
            pass
        try:
            new_atrs[prev_id] = graph.edge_attr[prev_id]
        except Exception:
            pass

    print(f'Double edges removed out of total {deleted}/{total_edges} ')
    
    return new_edges, new_weights, new_atrs

def LoadTestData():
     
    
    data = TUDataset("/Data","MUTAG",transform=NormalizeFeatures())[0]
    #data = KarateClub(transform=NormalizeFeatures())
    
    testdata = Data()
    new_edges, new_weights, new_atrs = RemoveDoubleEdges(data)
    
    testdata.edge_index = new_edges
    testdata.edge_weight = torch.rand(len(testdata.edge_index[0]))
    testdata.edge_attr = testdata.edge_weight.flatten()
    testdata.num_nodes = len(data.x) 
    testdata.num_edges = len(data.edge_index[0])
    nodefeats = torch.ones([testdata.num_nodes])
    testdata.node_features = nodefeats
    testdata.x = torch.ones([testdata.num_nodes,1])
    testdata = [testdata]
    original = [testdata[0].clone()]
    for g in testdata: g.num_edges = len(testdata[0].x)
    target = []
    converted_dataset = []
    
    for dataitem in testdata[:1]:
        print("Blossom matching")
        blossominput = []
        for i in range(len(dataitem.edge_index[0])):
            blossominput.append((dataitem.edge_index[0][i].item(),
                                 dataitem.edge_index[1][i].item(),
                                 dataitem.edge_weight[i].item()))

        target.append(maxWeightMatching(blossominput))
        line_graph = ToLineGraph(dataitem, dataitem.edge_attr, verbose = False)
        converted_dataset.append(line_graph)
    
    return original, converted_dataset, target
    
    testdata = Data()
    testdata.edge_index = torch.tensor([[0,1,2,3,3,5,6,7,8,8],
                                        [1,2,3,0,4,6,7,8,5,9]], dtype=torch.long)
    testdata.edge_attr = torch.tensor([0.8,0.2,0.8,0.2,0.2,0.8,0.2,0.8,0.2,0.2])
    testdata.num_nodes = 10 
    testdata.num_edges = 10
    testdata.node_features = torch.ones([testdata.num_nodes])
    testdata.x = torch.ones([testdata.num_nodes,1])
    
    
    original = testdata.clone()
    line_graph = ToLineGraph(testdata, testdata.edge_attr)
    
    target = [1,0,3,2,-1,6,5,8,7,-1]
    
    return [original], [line_graph], [target]


def LoadValExample():
        
    testdata = Data()
    testdata.edge_index = torch.tensor([[0,1,2,3,3],
                                        [1,2,3,0,4]], dtype=torch.long)
    testdata.edge_attr = torch.tensor([0.2,0.8,0.2,0.2,0.8])
    testdata.num_nodes = 5 
    testdata.num_edges = 5
    testdata.node_features = torch.ones([testdata.num_nodes])
    testdata.x = torch.ones([testdata.num_nodes,1])
    
    
    original = testdata.clone()
    line_graph = ToLineGraph(testdata, testdata.edge_attr)
    
    target = [-1,2,1,4,3]
    
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
        
        dataset = GNNBenchmarkDataset('/Data', datasetname, transform=NormalizeFeatures())[:count]
        mydataset = []
        for i, dataitem in enumerate(dataset): 
            dataitem.num_edges = len(dataitem.pos)
            new_edges, new_weights, new_atrs = RemoveDoubleEdges(dataitem)
            dataitem.edge_index = new_edges
            dataitem.edge_weight = torch.reshape(new_weights, (len(new_weights), 1))
            dataitem.edge_attr = new_atrs
            mydataset.append(dataitem)
        dataset = mydataset   
        return dataset, converted_dataset[:count], target[:count]
    
    except Exception:
        print(file_name, " not found creating new datafile")
        print("Downloading initial dataset")
        dataset = GNNBenchmarkDataset('/Data', datasetname, transform=NormalizeFeatures())[:count]
        mydataset = []
        for i, dataitem in enumerate(dataset): 
            dataitem.num_edges = len(dataitem.pos)
            new_edges, new_weights, new_atrs = RemoveDoubleEdges(dataitem)
            dataitem.edge_index = new_edges
            dataitem.edge_weight = torch.reshape(new_weights, (len(new_weights), 1))
            dataitem.edge_attr = new_atrs
            mydataset.append(dataitem)
        original = copy.deepcopy(mydataset)
        dataset = mydataset
        
        target = []
        converted_dataset = []
        
        for dataitem in dataset:
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
    
    return original, converted_dataset, target
    

def CountDegree(graph):
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes
    deg = [0] * num_nodes
    for i in range(len(edge_index[0])):
        from_node = edge_index[0][i].item()
        to_node = edge_index[1][i].item()
        deg[from_node]+=1
    return deg

def VisualizeConverted(graph):
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes
    
    s = "source target \n"
    with open("data/visualizationFile.csv", 'w') as file:
        for i in range(len(edge_index[0])):
            from_node = edge_index[0][i].item()
            to_node = edge_index[1][i].item()
            s += f'{from_node} {to_node} \n'
        file.writelines(s)    
    
    s = "id w label \n"
    with open("data/visualizationFileMetadata.csv", 'w') as file:
        for i in range(num_nodes):
            w = graph.x[i].item()
            y = graph.y[i]
            s += f'{i} {w:.4f} {y} \n'
        file.writelines(s)
        
def VisualizeOriginal(graph):
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes
    
    s = "source target w \n"
    with open("data/visualizationOriginalFile.csv", 'w') as file:
        for i in range(len(edge_index[0])):
            from_node = edge_index[0][i].item()
            to_node = edge_index[1][i].item()
            w = graph.edge_attr[i].item()
            s += f'{from_node} {to_node} {w}\n'
        file.writelines(s)