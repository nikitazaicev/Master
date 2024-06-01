import torch
from torch_geometric.transforms import NormalizeFeatures, ToUndirected, to_undirected
from torch_geometric.datasets import GNNBenchmarkDataset, KarateClub, TUDataset
from torch_geometric.data import Data
from torch.utils.data import TensorDataset
from torch.nn.functional import normalize
from blossom import maxWeightMatching
import LineGraphConverter as lgc
import pickle
import numpy as np
import copy
import ssgetpy as ss
import os
import GreedyPicker as gp
from scipy.io import mmread
import torch_geometric.transforms as T
from torchdata.datapipes.iter import IterableWrapper

np.random.seed(123)
torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def OverrideToGreedyBased(graphs):
    
    for g in graphs:
        target = gp.GreedyMatchingTargets(g)
        g.y = target
    
    return graphs

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

def TransformData():
    return T.Compose([NormalizeFeatures(),ToUndirected()])

def LoadTrain(datasetname='MNIST', skipLine=True, limit=0, doNormalize = False):
    if (not os.path.exists('data/train/target_data.pkl')
    or (not os.path.exists('data/train/converted_dataset.pkl') and not skipLine)): LoadData(datasetname,limit)
    file_name = 'data/train/target_data.pkl'
    with open(file_name, 'rb') as file:
        print(file_name, " loaded")
        target = pickle.load(file)
    
    if limit == 0: limit = len(target)
    if limit > 0: target = target[:limit]   
    
    dataset = GNNBenchmarkDataset('data', datasetname, split="train", transform=TransformData())[:limit]
    undirected = False
    dataset = PreproccessOriginal(dataset,target,datasetname,undirected,doNormalize)
    
    converted_dataset = 0
    if not skipLine:
        file_name = 'data/train/converted_dataset.pkl'
        with open(file_name, 'rb') as file:
            print(file_name, " loaded")
            converted_dataset = pickle.load(file)[:limit]
            
    return dataset, converted_dataset, target

def LoadVal(datasetname='MNIST', skipLine=True, limit=0, doNormalize = False):
    if not os.path.exists('data/val/target_data.pkl'): LoadData(datasetname, limit)
    
    file_name = 'data/val/target_data.pkl'
    with open(file_name, 'rb') as file:
        print(file_name, " loaded")
        target = pickle.load(file)
    
    if limit == 0: limit = len(target)
    if limit > 0: target = target[:limit]       
    
    dataset = GNNBenchmarkDataset('data', datasetname, split="val", transform=TransformData())[:limit] 
    
    dataset = PreproccessOriginal(dataset,target,datasetname,True,doNormalize)
    
    converted_dataset = 0
    if not skipLine:
        file_name = 'data/val/converted_dataset.pkl'
        with open(file_name, 'rb') as file:
            print(file_name, " loaded")
            converted_dataset = pickle.load(file)[:limit] 
    
    return dataset, converted_dataset, target

def LoadTest(datasetname='MNIST', limit=0, skipLine=True, doNormalize = False):
    if not os.path.exists('data/test/target_data.pkl'): LoadData(datasetname)
    file_name = 'data/test/target_data.pkl'
    with open(file_name, 'rb') as file:
        print(file_name, " loaded")
        target = pickle.load(file)        
    
    if limit == 0: limit = len(target)
    if limit > 0: target = target[:limit]    
    
    dataset = GNNBenchmarkDataset('data', datasetname, split="test", transform=TransformData())[:limit]
    dataset = PreproccessOriginal(dataset,target,datasetname,doNormalize)
    
    converted_dataset = []
    if not skipLine:
        file_name = 'data/test/converted_dataset.pkl'
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

def PreproccessOriginal(dataset, target, datasetname, undirected = True, doNormalize = False):
    print("Proccessing data")
    
    mydataset = []
    for i, dataitem in enumerate(dataset):                 
        
        if datasetname=='MNIST': 
            wmax = torch.max(dataitem.edge_attr, dim=0).values
            dataitem.edge_attr = dataitem.edge_attr / wmax
            dataitem.edge_attr = dataitem.edge_attr.to(device)
            dataitem.edge_weight = torch.reshape(dataitem.edge_attr, (len(dataitem.edge_attr), 1))
            dataitem.edge_weight = dataitem.edge_weight.to(device)
        dataitem.x, dataitem.adj = lgc.AugmentOriginalNodeFeatures(dataitem, undirected = undirected)
        dataitem.y = target[i]
        mydataset.append(dataitem)
    return mydataset

def ProccessData(dataset, datasetname, undirected = True, skipLine = True):
    mydataset = []
    print("Proccessing graphs, total = ", len(dataset))
    for i, dataitem in enumerate(dataset): 
        if datasetname=='MNIST': 
            wmax = torch.max(dataitem.edge_attr, dim=0).values
            dataitem.edge_attr = dataitem.edge_attr / wmax
            dataitem.edge_weight = torch.reshape(dataitem.edge_attr, 
                                                 (len(dataitem.edge_attr), 1))        
        dataitem.x, dataitem.adj = lgc.AugmentOriginalNodeFeatures(dataitem, undirected = undirected)

        mydataset.append(dataitem)
        if (i + 1) % 100 == 0: print(f'graph{i+1}/{len(dataset)}')
    original = copy.deepcopy(mydataset)
    dataset = mydataset

    target = []
    converted = []
    print("Blossom matching and line graph convertion")
    for idx, dataitem in enumerate(dataset):
        blossominput, uniqueEdges = [], set()
        for idx2 in range(len(dataitem.edge_index[0])):
            (f,t) = (dataitem.edge_index[0][idx2].item(), dataitem.edge_index[1][idx2].item())
            w = dataitem.edge_weight[idx2][0].item()
            
            if (f,t) not in uniqueEdges and (t,f) not in uniqueEdges:
                blossominput.append((f,t,w))
                uniqueEdges.add((f,t))
                uniqueEdges.add((t,f))

        try:
            targetClasses = AssignTargetClasses(dataitem, maxWeightMatching(blossominput))
        except: 
            print("blossom failed skipping")
            continue
        target.append(targetClasses)
        dataitem.y = targetClasses
        original[idx].y = targetClasses
        line_graph = 0
        if not skipLine: 
            line_graph = lgc.ToLineGraph(dataitem, verbose = False)
        converted.append(line_graph) 
        if (idx + 1) % 10 == 0: print(f'graph{idx+1}/{len(dataset)}')                   
    
    return original, converted, target   

def LoadData(datasetname='MNIST', limit=100):
    print("LOADING DATASETS")
    print("-------------------")
    transform = TransformData()
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


def LoadDataCustom(limit=0):
    print("LOADING DATASETS")
    print("-------------------")
    print("LOADING DATA")

    file_name = 'data/custom/customdatasetfiles.pkl'
    with open(file_name, 'rb') as file:
        print(file_name, " loaded")
        filenames = pickle.load(file)
    if limit == 0: limit = len(filenames)     
    print("LOADING AND PROCCESSING DATA")
    data = []
    count = 0
    for filename in filenames[:limit]:
        mmformat = mmread(f'data/custom/{filename}/{filename}.mtx').toarray()
        
        # file_name = f'data/customtrain/weights/{filename}.pkl'
        # with open(file_name, 'rb') as file:
        #     #print(file_name, " loaded")
        #     ws = pickle.load(file)
        
        graph, reason = FromMMformat(mmformat)
        if reason != "OK": 
            print("Skiped ", filename, reason)
            continue
        if "cage" in filename or "California" in filename or "G22" in filename or "G55" in filename or graph.num_edges > 300000 or graph.num_nodes > 10000: 
            print("Skiped ", filename)
            data.append(None)
        else: data.append(graph)
        count += 1
        if (count) % 50 == 0: print(f'graph{count}/{len(filenames[:limit])}')
        
    
    if os.path.exists('data/customtrain/target_data.pkl'):
        file_name = 'data/customtrain/target_data.pkl'
        with open(file_name, 'rb') as file:
            print(file_name, " loaded")
            targetset = pickle.load(file)
        
        target, converted, dataset = [],[],[]
        for i, g in enumerate(data):
              if g is not None:
                  target.append(targetset[i])
                  dataset.append(data[i])
        dataset = PreproccessOriginal(dataset, target,  "custom", undirected = True, doNormalize=False)     
        print("DONE")
        assert(len(dataset)==len(target))
        for i, d in enumerate(dataset):
            assert(len(d.y) == len(target[i]))
        
        return dataset, converted, target
    
    dataset, converted, target = ProccessData(data, "custom", undirected = True, skipLine=True)     
    SaveCustom(target, converted)
    assert(len(dataset)==len(target))
    print("DONE AND SAVED")
    
    return dataset,converted,target


def SaveCustom(test_target, test_converted):                   
    file_name = 'data/customtrain/target_data.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(test_target, file)
        print(f'Object successfully saved to "{file_name}"')
        
    file_name = 'data/customtrain/converted_dataset.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(test_converted, file)
        print(f'Object successfully saved to "{file_name}"')
    return 

def AssignTargetClasses(graph, target):
    classes, uniqueEdges = [], set()
    for j in range(graph.num_edges):
        from_node = graph.edge_index[0][j].item()
        to_node = graph.edge_index[1][j].item()

        #if (from_node,to_node) in uniqueEdges: continue

        if target[from_node] == to_node or target[to_node] == from_node:
            classes.append(1)
        else:
            classes.append(0)
        #uniqueEdges.add((to_node,from_node))
        #uniqueEdges.add((from_node,to_node))
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
        
def FromMMformat(graph, ws = None):
    original_graph = Data()
    from_nodes, to_nodes, new_weights = [],[],[]
    j, count = 0, 0
    nodeMap = dict()

    for row in range(len(graph)-1):    
        for col in range(j):
            w = graph[row][col]
            if w == 0 or None: continue
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
    
    if original_graph.num_nodes == 0: original_graph, " original_graph.num_nodes == 0"
    if original_graph.num_edges == 0: original_graph, "original_graph.num_edges == 0"
    if len(new_weights)==0: return original_graph, "(new_weights)==0"
    new_weights = torch.Tensor(new_weights).flatten()
    if 0 == torch.max(new_weights, dim=0).values: return original_graph, "0 == torch.max(new_weights, dim=0).values"
    
    assert(len(from_nodes)==len(to_nodes))
    dir1 = torch.Tensor([from_nodes,to_nodes]).type(torch.int64)
    dir2 = torch.Tensor([to_nodes,from_nodes]).type(torch.int64)
    original_graph.edge_index = torch.cat((dir1, dir2),1)

    if ws is None:
        wmin = torch.min(new_weights, dim=0).values
        wmax = torch.max(new_weights, dim=0).values
        if wmin < 0: new_weights -= (wmin-1)
        wmax = torch.max(new_weights, dim=0).values
        if wmin == wmax or wmax == 0: 
            new_weights = torch.rand((len(new_weights),1))
        new_weights = new_weights / wmax
        #new_weights = new_weights * 10
        new_weights = torch.cat((new_weights, new_weights),0)
    else: 
        new_weights = ws  / 10
    
    original_graph.edge_attr = new_weights.flatten()
    new_weights = torch.reshape(new_weights, (len(new_weights),1))
    original_graph.edge_weight = new_weights 
    original_graph.num_nodes = count
    original_graph.num_edges = len(original_graph.edge_index[0])
    
    return original_graph, "OK"

def GreedyBad(graph):
    print("Mutating for greedy")
    dp = IterableWrapper(range(int(len(graph.edge_weight)/2))).shuffle()
    mutated = set()
    adjList = GenerateAdjList(graph)
    for i in dp:
        if i in mutated: continue
        val = 1.0
        graph.edge_weight[i] = val 
        graph.edge_attr[i] = val 
        graph.edge_weight[i+int(len(graph.edge_weight)/2)] = val 
        graph.edge_attr[i+int(len(graph.edge_weight)/2)] = val 
        mutated.add(i)
        mutated.add(i+int(len(graph.edge_weight)/2))
        
        (f,t) = (graph.edge_index[0][i].item(), graph.edge_index[1][i].item())

        for (n,w,idx) in adjList[f].union(adjList[t]):
            if idx in mutated or idx >= int(len(graph.edge_weight)/2): continue
            val = 0.99
            graph.edge_weight[idx] = val
            graph.edge_attr[idx] = val
            graph.edge_weight[idx+int(len(graph.edge_weight)/2)] = val
            graph.edge_attr[idx+int(len(graph.edge_weight)/2)] = val
            mutated.add(idx)
            mutated.add(idx+int(len(graph.edge_weight)/2))
            
            (f2,t2) = (graph.edge_index[0][idx].item(), graph.edge_index[1][idx].item())
            for (n2,w2,idx2) in adjList[f2].union(adjList[t2]):
                if idx2 in mutated or idx2 >= int(len(graph.edge_weight)/2): continue
                val = 0.001
                graph.edge_weight[idx2] = val
                graph.edge_attr[idx2] = val
                graph.edge_weight[idx2+int(len(graph.edge_weight)/2)] = val
                graph.edge_attr[idx2+int(len(graph.edge_weight)/2)] = val
                mutated.add(idx2)
                mutated.add(idx2+int(len(graph.edge_weight)/2))
    print("Mutating done")                
    return graph

def GenerateAdjList(graph):
    adjL = [set() for _ in range(graph.num_nodes)]
    for i in range(len(graph.edge_index[0])):
        w = graph.edge_weight[i].item()
        from_node = graph.edge_index[0][i].item()
        to_node = graph.edge_index[1][i].item()
        adjL[from_node].add((to_node,w,i))
        adjL[to_node].add((from_node,w,i))
    return adjL

def LoadValGoodCase(filenames = []):
        
    dataset = []
    converted_dataset = []
    target = []
    for filename in filenames:
        mmformat = mmread(filename).toarray()
        original_graph = FromMMformat(mmformat)
        line_graph = lgc.ToLineGraph(FromMMformat(mmformat), verbose = False)
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
