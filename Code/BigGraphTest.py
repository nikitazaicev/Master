import pickle
import MyGCN
import DataLoader as dl
import torch
import random
import GreedyPicker as gp
import time
from blossom import maxWeightMatching
import LineGraphConverter as lgc
import ReductionsManager as rm
import ssgetpy as ss
from scipy.io import mmread
torch.manual_seed(123)
random.seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current CUDA version: ", torch.version.cuda, "\n")
print("TEST BIG GRAPH")

# DATASET CANDIDATES
# webbase-2001
# it-2004
# GAP-twitter 
# twitter7
# GAP-web
# sk-2005

print("----------------")

try:
    with open('data/MyModel.pkl', 'rb') as file:
        print("Loading Model")
        model = pickle.load(file).to(device)
    with open('data/MyModelClass.pkl', 'rb') as file:
        classifier = pickle.load(file).to(device) 
        modelLoaded = True
except Exception: print("No model found. EXIT")


graphs, converted_dataset, target = dl.LoadTest(limit=10)

gr = 'vanHeukelum'
dataset = ss.search(group = gr, 
                    kind='Directed Weighted Graph', 
                    limit = 1, 
                    rowbounds = (10000,50000),
                    colbounds = (10000,50000))
filenames = []
for dataitem in dataset:
    filenames.append(dataitem.name)

matrices = dataset.download(destpath=f'data/{gr}', extract=True)
dataset, converted_dataset, resultsGNN, resultsGreedy, resultsOpt = [],[],[],[],[]
for filename in filenames:
    print("Current graph:", filename)
    mmformat = mmread(f'data/{gr}/{filename}/{filename}.mtx').toarray()
    graph = dl.FromMMformat(mmformat)
    graph.edge_weight = torch.reshape(graph.edge_weight, (len(graph.edge_weight), 1))
    print("EDGES: ", len(graph.edge_index[0]))
    if graph.edge_weight is None: graph.edge_weight = graph.edge_attr
    print("EDGE ATTRS : ", graph.edge_attr)
    print("EDGE WEIGHTS : ", graph.edge_weight)
    
    blossominput = []
    target = []
    for i in range(len(graph.edge_index[0])):
        blossominput.append((graph.edge_index[0][i].item(),
                              graph.edge_index[1][i].item(),
                              graph.edge_attr[i].item()))
    start_time = time.time()
    print("started blossom matching")
    match=maxWeightMatching(blossominput)
    timeTotal = time.time() - start_time
    print("done greedy matching")    
    target.append(match)
    
    
    print("started weight count")
    val_y = target
    val_y_item = torch.LongTensor(val_y[0])
    classes = []
    for j in range(len(graph.edge_index[0])):
        from_node = graph.edge_index[0][j].item()
        to_node = graph.edge_index[1][j].item() 
        if val_y_item[from_node] == to_node:
            classes.append(1)
        else:
            classes.append(0)
    val_y_item = torch.FloatTensor(classes)
    true_edges_idx = (val_y_item == 1).nonzero(as_tuple=True)[0]
    totalWeightOpt = 0
    for idx in true_edges_idx:
        from_node = graph.edge_index[0][idx].item()
        to_node = graph.edge_index[1][idx].item()
        totalWeightOpt += graph.edge_attr[idx]
    print("done weight count")
        
    resultsOpt.append([("TIME", timeTotal),("WEIGHT", totalWeightOpt.item())])

    weightSum = 0
    print("Before reduction: ", graph.num_nodes)
    graph.x = lgc.AugmentOriginalNodeFeatures(graph)
    graph, weightRed = rm.ApplyReductionRules(graph)
    weightSum += weightRed
    print("After reduction: ", graph.num_nodes)
    start_time = time.time()
    print("started greedy matching")
    weightGreedy, pickedEdgeIndeces = gp.GreedyMatchingOrig(graph)
    print("done greedy matching")
    timeTotal = time.time() - start_time
    resultsGreedy.append([("TIME", timeTotal),("WEIGHT", weightGreedy)])
    
    
    print("started gnn matching")
    graph = graph.to(device)
    start_time = time.time()
    graph, weightGnn = MyGCN.GNNMatching(model, classifier, graph.to(device))
    weightRes, pickedEdgeIndeces = gp.GreedyMatchingOrig(graph)
    weightSum += weightGnn
    weightSum += weightRes
    weightResDif = weightRes/weightSum 
    timeTotal = time.time() - start_time
    resultsGNN.append([("TIME", timeTotal),("WEIGHT", weightSum.item()),("Gnn Res weight Diff", weightResDif.item())])
    print("done gnn matching")
    
avgTimeGnn, avgWeightGnn = 0,0
avgTimeGreed, avgWeightGreed = 0,0
avgTimeOpt, avgWeightOpt = 0,0
weightResDif = 0
for idx, r in enumerate(resultsGNN):
    print("GNN = ", resultsGNN[idx])
    print("GREED = ", resultsGreedy[idx])
    print("----------------")
    avgTimeGnn += resultsGNN[idx][0][1]
    avgWeightGnn += resultsGNN[idx][1][1]
    avgTimeGreed += resultsGreedy[idx][0][1]
    avgWeightGreed += resultsGreedy[idx][1][1]
    avgTimeOpt += resultsOpt[idx][0][1]
    avgWeightOpt += resultsOpt[idx][1][1]
    weightResDif += resultsGNN[idx][2][1]
    
    

print("TIME")
print("Opt AVG Time = ", avgTimeOpt/len(resultsGNN))
print("GNN AVG Time = ", avgTimeGnn/len(resultsGNN))
print("GREED AVG Time = ", avgTimeGreed/len(resultsGNN))
print("----------------")
print("WEIGHT")
print("Opt AVG Weight = ", avgWeightOpt/len(resultsGNN))
print("GNN AVG Weight = ", avgWeightGnn/len(resultsGNN))
print("GREED AVG Weight = ", avgWeightGreed/len(resultsGNN))
print("----------")
print("weightResDif = ", (weightResDif/len(resultsGNN)))
print("-----------")
print("END")    
    