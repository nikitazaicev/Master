import torch
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import GNNBenchmarkDataset, KarateClub, TUDataset
from torch_geometric.data import Data
from blossom import maxWeightMatching
import LineGraphConverter as lgc
import pickle
import numpy as np
import copy
import ssgetpy as ss
import os
from scipy.io import mmread
import torch_geometric.transforms as T

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
            new_weights[prev_id] = graph.edge_attr[prev_id]
        except Exception:
            pass

    return new_edges, new_weights, new_atrs

def WeightsExperiment(graph):    
    graphOnes, graphOneTwos = copy.deepcopy(graph), copy.deepcopy(graph)
    size = graph.edge_weight.size()
    graphOnes.edge_weight = torch.ones(size,dtype=torch.float)
    graphOneTwos.edge_weight = torch.randint(1,3,size,dtype=torch.float)
    graphs = [graph, graphOnes, graphOneTwos]
    targets = []
    for g in graphs:
        blossominput = []
        for i in range(len(g.edge_index[0])):
            blossominput.append((g.edge_index[0][i].item(),
                                 g.edge_index[1][i].item(),
                                 g.edge_weight[i][0].item()))

        targetClasses = AssignTargetClasses(g, maxWeightMatching(blossominput))
        targets.append(targetClasses)
        g.y = targetClasses

    return graphs, targets

def LoadTrain(datasetname='MNIST', skipLine=True, limit=0):
    if not os.path.exists('data/train/target_data.pkl'): LoadData(datasetname,limit)
    file_name = 'data/train/target_data.pkl'
    with open(file_name, 'rb') as file:
        print(file_name, " loaded")
        target = pickle.load(file)
    
    if limit == 0: limit = len(target)
    if limit > 0: target = target[:limit]   
    
    transform = T.Compose([NormalizeFeatures()])
    dataset = GNNBenchmarkDataset('data', datasetname,  split="train", transform=transform)[:limit]
    dataset = PreproccessOriginal(dataset,target,datasetname)
    
    converted_dataset = 0
    if not skipLine:
        file_name = 'data/train/converted_dataset.pkl'
        with open(file_name, 'rb') as file:
            print(file_name, " loaded")
            converted_dataset = pickle.load(file)[:limit]
            
    return dataset, converted_dataset, target

def LoadVal(datasetname='MNIST', skipLine=True, limit=0):
    if not os.path.exists('data/val/target_data.pkl'): LoadData(datasetname)
    file_name = 'data/val/target_data.pkl'
    with open(file_name, 'rb') as file:
        print(file_name, " loaded")
        target = pickle.load(file)
    
    if limit == 0: limit = len(target)
    if limit > 0: target = target[:limit]       
    
    transform = T.Compose([NormalizeFeatures()])
    dataset = GNNBenchmarkDataset('data', datasetname, split="val", transform=transform)[:limit] 
    dataset = PreproccessOriginal(dataset,target,datasetname)
    
    converted_dataset = 0
    if not skipLine:
        file_name = 'data/train/converted_dataset.pkl'
        with open(file_name, 'rb') as file:
            print(file_name, " loaded")
            converted_dataset = pickle.load(file)[:limit] 
    
    return dataset, converted_dataset, target

def LoadTest(datasetname='MNIST', limit=0, skipLine=True):
    if not os.path.exists('data/test/target_data.pkl'): LoadData(datasetname)
    file_name = 'data/test/target_data.pkl'
    with open(file_name, 'rb') as file:
        print(file_name, " loaded")
        target = pickle.load(file)        
    
    if limit == 0: limit = len(target)
    if limit > 0: target = target[:limit]    
    
    transform = T.Compose([NormalizeFeatures()])
    dataset = GNNBenchmarkDataset('data', datasetname, split="test", transform=transform)[:limit]
    dataset = PreproccessOriginal(dataset,target,datasetname)
    
    converted_dataset = []
    if not skipLine:
        file_name = 'data/train/converted_dataset.pkl'
        with open(file_name, 'rb') as file:
            print(file_name, " loaded")
            converted_dataset = pickle.load(file)[:limit]
            
    return dataset, converted_dataset, target

def SaveTrain(train_target, train_converted):    
    file_name = 'data/train/target_data.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(train_target, file)
        print(f'Object successfully saved to "{file_name}"')
        
    file_name = 'data/train/converted_dataset.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(train_converted, file)
        print(f'Object successfully saved to "{file_name}"')
    return

def SaveVal(val_target, val_converted):             
    file_name = 'data/val/target_data.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(val_target, file)
        print(f'Object successfully saved to "{file_name}"')
        
    file_name = 'data/val/converted_dataset.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(val_converted, file)
        print(f'Object successfully saved to "{file_name}"')
    return

def SaveTest(test_target, test_converted):                   
    file_name = 'data/test/target_data.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(test_target, file)
        print(f'Object successfully saved to "{file_name}"')
        
    file_name = 'data/test/converted_dataset.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(test_converted, file)
        print(f'Object successfully saved to "{file_name}"')
    return 

def PreproccessOriginal(dataset, target, datasetname):
    print("Proccessing data")
    mydataset = []
    for i, dataitem in enumerate(dataset): 
        new_edges, new_weights, new_atrs = RemoveDoubleEdges(dataitem)
        dataitem.num_edges = len(new_edges[0])
        dataitem.edge_index = new_edges
        dataitem.edge_weight = torch.reshape(new_weights, (len(new_weights), 1))
        dataitem.edge_attr = new_atrs
        dataitem.x = lgc.AugmentOriginalNodeFeatures(dataitem)
        dataitem.y = target[i]
        if datasetname=='MNIST': dataitem.edge_weight = torch.reshape(new_atrs, (len(new_atrs), 1))
        mydataset.append(dataitem)
    original = copy.deepcopy(mydataset) 
    return original

def ProccessData(dataset, datasetname):
    mydataset = []
    print("Proccessing graphs, total = ", len(dataset))
    for i, dataitem in enumerate(dataset): 
        new_edges, new_weights, new_atrs = RemoveDoubleEdges(dataitem)
        dataitem.num_edges = len(new_edges[0])
        dataitem.edge_index = new_edges
        dataitem.edge_weight = torch.reshape(new_weights, (len(new_weights), 1))
        dataitem.edge_attr = new_atrs
        dataitem.x = lgc.AugmentOriginalNodeFeatures(dataitem)
        if datasetname=='MNIST': dataitem.edge_weight = torch.reshape(new_atrs, (len(new_atrs), 1))
        mydataset.append(dataitem)
        if (i + 1) % 10 == 0: print(f'graph{i+1}/{len(dataset)}')
    original = copy.deepcopy(mydataset)
    dataset = mydataset

    target = []
    converted = []
    
    print("Blossom matching")
    for idx, dataitem in enumerate(dataset):
        blossominput = []
        for i in range(len(dataitem.edge_index[0])):
            blossominput.append((dataitem.edge_index[0][i].item(),
                                 dataitem.edge_index[1][i].item(),
                                 dataitem.edge_attr[i].item()))

        targetClasses = AssignTargetClasses(dataitem, maxWeightMatching(blossominput))
        target.append(targetClasses)
        original[idx].y = targetClasses
        line_graph = lgc.ToLineGraph(dataitem, dataitem.edge_attr, verbose = False)
        converted.append(line_graph) 
        if (idx + 1) % 10 == 0: print(f'graph{idx+1}/{len(dataset)}')                   
    
    return original, converted, target   

def LoadData(datasetname='MNIST', limit=0):
    print("LOADING DATASETS")
    print("-------------------")
    transform = T.Compose([NormalizeFeatures()])
    print("LOADING TRAINING DATA")
    dataset = GNNBenchmarkDataset('data', datasetname, split="train", transform=transform)
    if limit>0: dataset = dataset[:limit]
    original, converted, target = ProccessData(dataset, datasetname)              
    SaveTrain(target, converted)
    print("SAVED TRAINING DATA")
    print("-------------------")
    print("LOADING VALIDATION DATA")
    dataset = GNNBenchmarkDataset('data', datasetname, split="val", transform=transform)
    if limit>0: dataset = dataset[:limit]
    original, converted, target = ProccessData(dataset, datasetname)              
    SaveVal(target, converted)
    print("SAVED VALIDATION DATA")
    print("-------------------")
    print("LOADING TEST DATA")
    dataset = GNNBenchmarkDataset('data', datasetname, split="test", transform=transform)
    if limit>0: dataset = dataset[:limit]
    original, converted, target = ProccessData(dataset, datasetname)              
    SaveTest(target, converted)
    print("SAVED TEST DATA")
    print("-------------------")
    return 

def AssignTargetClasses(graph, target):
    classes = [] 
    for j in range(graph.num_edges):
        from_node = graph.edge_index[0][j].item()
        to_node = graph.edge_index[1][j].item()
        if target[from_node] == to_node:
            classes.append(1)
        else:
            classes.append(0)
    y = torch.LongTensor(classes).to(device)
    graph.y = y
    return y


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
    
    s = "source target w \n"
    with open("data/visualizationOriginalFile.csv", 'w') as file:
        for i in range(len(edge_index[0])):
            from_node = edge_index[0][i].item()
            to_node = edge_index[1][i].item()
            w = graph.edge_attr[i].item()
            s += f'{from_node} {to_node} {w}\n'
        file.writelines(s)
        
def FromMMformat(graph):
    original_graph = Data()
    from_nodes, to_nodes, new_weights = [],[],[]
    minW, j, count = 0, 0, 0
    nodeMap = dict()
    
    for row in range(len(graph)-1):    
        for col in range(j):
            w = graph[row][col]
            if w == 0: continue
            if w < minW: minW = w
            
            if row in nodeMap: 
                from_nodes.append(nodeMap[row])
            else: 
                from_nodes.append(count)
                nodeMap[row] = count
                count+=1
            
            if col in nodeMap: 
                to_nodes.append(nodeMap[col])
            else: 
                to_nodes.append(count)
                nodeMap[col] = count
                count+=1
                
            new_weights.append(graph[row][col])            
        if j < len(graph[0]): j += 1    
    
    if minW >= 0: minW = 0 
    assert(len(from_nodes)==len(to_nodes))
    original_graph.edge_index = torch.Tensor([from_nodes,to_nodes]).type(torch.int64)
    original_graph.edge_weight = torch.add(torch.Tensor(new_weights), -1*minW)
    original_graph.edge_attr = original_graph.edge_weight
    original_graph.num_nodes = count
    original_graph.num_edges = len(from_nodes)
    return original_graph



def LoadValGoodCase(filenames = []):
        
    dataset = []
    converted_dataset = []
    target = []
    for filename in filenames:
        mmformat = mmread(filename).toarray()
        original_graph = FromMMformat(mmformat)
        line_graph = lgc.ToLineGraph(FromMMformat(mmformat), original_graph.edge_attr, verbose = False)
        converted_dataset.append(line_graph)
        original_graph.x = lgc.AugmentOriginalNodeFeatures(original_graph)
        dataset.append(original_graph)

        blossominput = []
        for i in range(len(original_graph.edge_index[0])):
            blossominput.append((original_graph.edge_index[0][i].item(),
                                 original_graph.edge_index[1][i].item(),
                                 original_graph.edge_attr[i].item()))
        target.append(maxWeightMatching(blossominput))
        assert((len(original_graph.edge_index[0])==line_graph.num_nodes)) 
    return dataset, converted_dataset, target

def LoadValExample():
        
    dataset = ss.search(group = 'HB')
    filenames = [] 
    for dataitem in dataset:
        filenames.append(dataitem.name)

    dataset.download(destpath="data/HB",extract=True)
    dataset = []
    converted_dataset = []
    target = []
    for filename in filenames:
        mmformat = mmread(f'data/HB/{filename}/{filename}.mtx').toarray()
        original_graph = FromMMformat(mmformat)
        line_graph = lgc.ToLineGraph(FromMMformat(mmformat), original_graph.edge_attr, verbose = False)
        converted_dataset.append(line_graph)
        dataset.append(original_graph)
    
        blossominput = []
        for i in range(len(original_graph.edge_index[0])):
            blossominput.append((original_graph.edge_index[0][i].item(),
                                 original_graph.edge_index[1][i].item(),
                                 original_graph.edge_attr[i].item()))
        target.append(maxWeightMatching(blossominput))
        assert((len(original_graph.edge_index[0])==line_graph.num_nodes)) 
    return dataset, converted_dataset, target