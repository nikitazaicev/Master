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
        except Exception:
            pass

    return new_edges, new_weights, new_atrs

def LoadTestData():
     
    #datas = KarateClub(transform=NormalizeFeatures())
    #datas = TUDataset("/Data", "MUTAG", transform=NormalizeFeatures())
    datas = TUDataset("/Data", "REDDIT-BINARY", transform=NormalizeFeatures())[:10]
    datalist = []
    original = []
    for data in datas:
               
        testdata = Data()
        new_edges, new_weights, new_atrs = RemoveDoubleEdges(data)
        
        testdata.edge_index = new_edges
        testdata.edge_weight = torch.rand(len(testdata.edge_index[0]))
        testdata.edge_attr = testdata.edge_weight.flatten()
        testdata.num_nodes = data.num_nodes#len(data.x) 
        testdata.num_edges = len(testdata.edge_index[0])
        nodefeats = torch.ones([testdata.num_nodes])
        testdata.node_features = nodefeats
        testdata.x = torch.ones([testdata.num_nodes,1])
        original.append(testdata.clone())
        datalist.append(testdata)
    
    #for data in datas: data.num_edges = len(data.x)
    target = []
    converted_dataset = []
    
    print("Blossom matching")
    for dataitem in datalist:
        blossominput = []
        for i in range(len(dataitem.edge_index[0])):
            blossominput.append((dataitem.edge_index[0][i].item(),
                                 dataitem.edge_index[1][i].item(),
                                 dataitem.edge_weight[i].item()))
        target.append(maxWeightMatching(blossominput))
        line_graph = lgc.ToLineGraph(dataitem, dataitem.edge_attr, verbose = False)
        converted_dataset.append(line_graph)
        
    assert(original[0].num_edges==len(original[0].edge_index[0]))
    return original, converted_dataset, target


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
        
        transform = T.Compose([NormalizeFeatures()])
        dataset = GNNBenchmarkDataset('data', datasetname, transform=transform)[:count]
        
        mydataset = []
        for i, dataitem in enumerate(dataset): 
            new_edges, new_weights, new_atrs = RemoveDoubleEdges(dataitem)
            dataitem.edge_index = new_edges
            dataitem.num_edges = len(new_weights)
            dataitem.edge_weight = torch.reshape(new_weights, (len(new_weights), 1))
            dataitem.edge_attr = new_atrs
            dataitem.x = lgc.AugmentNodeFeatures(dataitem)
            
            if datasetname=='MNIST': dataitem.edge_weight = torch.reshape(new_atrs, (len(new_atrs), 1))
            mydataset.append(dataitem)
        dataset = mydataset   
        return dataset, converted_dataset[:count], target[:count]
    
    except Exception:
        print(file_name, " not found creating new datafile")
        print("Downloading initial dataset")
        transform = T.Compose([NormalizeFeatures()])
        dataset = GNNBenchmarkDataset('data', datasetname, transform=transform)[:count]
        mydataset = []
        for i, dataitem in enumerate(dataset): 
            new_edges, new_weights, new_atrs = RemoveDoubleEdges(dataitem)
            dataitem.num_edges = len(new_edges[0])
            dataitem.edge_index = new_edges
            dataitem.edge_weight = torch.reshape(new_weights, (len(new_weights), 1))
            dataitem.edge_attr = new_atrs
            dataitem.x = lgc.AugmentNodeFeatures(dataitem)
            if datasetname=='MNIST': dataitem.edge_weight = torch.reshape(new_atrs, (len(new_atrs), 1))
            mydataset.append(dataitem)
        original = copy.deepcopy(mydataset)
        dataset = mydataset
        
        target = []
        converted_dataset = []
        
        
        print("Blossom matching")
        for dataitem in dataset:
            blossominput = []
            for i in range(len(dataitem.edge_index[0])):
                blossominput.append((dataitem.edge_index[0][i].item(),
                                     dataitem.edge_index[1][i].item(),
                                     dataitem.edge_attr[i].item()))

            target.append(maxWeightMatching(blossominput))
            line_graph = lgc.ToLineGraph(dataitem, dataitem.edge_attr, verbose = False)
            converted_dataset.append(line_graph)
        
        with open(file_name, 'wb') as file:
            pickle.dump(target, file)
            print(f'Object successfully saved to "{file_name}"')
            
        file_name = 'data/converted_dataset.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(converted_dataset, file)
            print(f'Object successfully saved to "{file_name}"')
    
    return original, converted_dataset, target    

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
        original_graph.x = lgc.AugmentNodeFeatures(original_graph)
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