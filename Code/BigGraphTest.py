import pickle
import MyGCN
import MyDataLoader as dl
import torch
import random
import GreedyPicker as gp
import time
from blossom import maxWeightMatching
import LineGraphConverter as lgc
import ReductionsManager as rm
import ssgetpy as ss
from scipy.io import mmread
import copy
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


try:
    #with open('data/MNISTTRAINED/MyModel.pkl', 'rb') as file:
    #with open('data/CUSTOMTRAINED/MyModel.pkl', 'rb') as file:
    with open('data/tempMyModel.pkl', 'rb') as file:
        print("Loading Model")
        model = pickle.load(file).to(device)
        model.eval()
    #with open('data/MNISTTRAINED/MyModelClass.pkl', 'rb') as file:
    #with open('data/CUSTOMTRAINED/MyModelClass.pkl', 'rb') as file:
    with open('data/tempMyModelClass.pkl', 'rb') as file:
        classifier = pickle.load(file).to(device) 
        classifier.eval()
        modelLoaded = True
except Exception: print("No model found. EXIT")

#graphs, converted_dataset, target = dl.LoadTest(limit=10)

#gr = 'Gset'
gr = 'Pajek'
#gr = 'Newman'
#gr = 'vanHeukelum'
dataset = ss.search(group = gr, 
                    #kind='Directed Weighted Graph', 
                    limit = 1, 
                    rowbounds = (2000,100000),
                    colbounds = (2000,100000))

useReduction = False
filenames = []
for dataitem in dataset:
    filenames.append(dataitem.name)

matrices = dataset.download(destpath=f'data/{gr}', extract=True)
dataset, converted_dataset, resultsGNN, resultsGreedy, resultsOpt = [],[],[],[],[]
for filename in filenames:
    print("Current graph:", filename)

    mmformat = mmread(f'data/{gr}/{filename}/{filename}.mtx').toarray()
    graph = dl.FromMMformat(mmformat)
    graph = graph.to(device)
    graph.edge_weight = torch.reshape(graph.edge_weight, (len(graph.edge_weight), 1))
    weightSum = 0
    
    weightRed, reductionImpact = 0, 0
    if useReduction: 
        print("Before reduction: ", graph.num_nodes)
        graph2, weightRed, reductionNodeIds = rm.ApplyReductionRules(copy.deepcopy(graph))
        weightSum += weightRed
        print(weightRed) 
        print("After reduction: ", graph2.num_nodes)

    graph = graph.to(device)
    print("EDGES: ", len(graph.edge_index[0]))
    print("NODES: ", graph.num_nodes)
    edgesnum, nodesnum = graph.num_edges, graph.num_nodes
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
    print("done blossom matching")    
    target.append(match)
    
    
    print("started weight count")
    val_y_item = torch.LongTensor(target[0])
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
    
    totalWeightOpt, duplicates = 0, set()
    for idx in true_edges_idx:
        from_node = graph.edge_index[0][idx].item()
        to_node = graph.edge_index[1][idx].item()
        if (from_node,to_node) in duplicates or (to_node,from_node) in duplicates: continue
        duplicates.add((from_node,to_node))
        duplicates.add((to_node,from_node))
        totalWeightOpt += graph.edge_attr[idx]
    print("done weight count")
        
    resultsOpt.append([("TIME", timeTotal),("WEIGHT", totalWeightOpt.item())])


    print("started gnn matching")
    start_time = time.time()
    graph.x, graph.adj = lgc.AugmentOriginalNodeFeatures(graph, undirected = True)
    print("NODE ATTR: ", graph.x[0])
    timeAugment = time.time() - start_time
    
    start_time = time.time()
    print("started greedy matching")
    weightGreedy, pickedEdgeIndeces = gp.GreedyMatchingOrig(graph)
    weightGreedy += weightRed
    print("GREEEEDY", weightGreedy)
    print("done greedy matching")
    timeTotal = time.time() - start_time
    resultsGreedy.append([("TIME", timeTotal),("WEIGHT", weightGreedy)])
    
    start_time = time.time()
    graph, weightGnn, pickedNodes, iterations = MyGCN.GNNMatching(model, classifier, graph.to(device), 0.50, 0.0, test = reductionNodeIds)
    
    print("GNN", weightGnn)
    print("GNN iters", iterations)
    weightRes, pickedEdgeIndeces = gp.GreedyMatchingOrig(graph)
    print("GNN REMAINDER", weightRes)
    
    weightSum += weightGnn
    weightSum += weightRes
    weightResDif = weightRes/weightSum
    reductionImpact = weightRed/weightSum  
    timeTotal = timeAugment + (time.time() - start_time)
    resultsGNN.append([("TIME", timeTotal),
                       ("WEIGHT", weightSum),
                       ("Gnn Res weight remainder", weightResDif),
                       ("Reduction weight portion", reductionImpact),
                       ("Gnn Dropped Nodes", len(pickedNodes))])
    print("done gnn matching")
    print("Current graph:", filename)
    print("Nodes = ", nodesnum, "Edges = ", edgesnum)
    
    
avgTimeGnn, avgWeightGnn = 0,0
avgTimeGreed, avgWeightGreed = 0,0
avgTimeOpt, avgWeightOpt = 0,0
weightResDif, droppedNodes = 0, 0
reductionImpact = 0
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
    reductionImpact += resultsGNN[idx][3][1]
    droppedNodes += resultsGNN[idx][4][1]
    
if useReduction: print("using reduction")
else: print("without reduction")
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
remainder = (weightResDif/len(resultsGNN))
print("remainder = ", remainder )
reductionImpact = (reductionImpact/len(resultsGNN))
print("reduction Impact = ", reductionImpact)
print("gnn impact = ", 1-reductionImpact-remainder)
print("Dropped Nodes = ", (droppedNodes/len(resultsGNN)))
print("-----------")
print("END")    
    