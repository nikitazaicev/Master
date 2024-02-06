import pickle
import MyGCN
import DataLoader as dl
from scipy.io import mmread
import torch
import random
import GreedyPicker as gp
import time
from blossom import maxWeightMatching
import LineGraphConverter as lgc
import ReductionsManager as rm
import ssgetpy as ss

torch.manual_seed(123)
random.seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current CUDA version: ", torch.version.cuda, "\n")
print("SEARCHING FOR SUITABLE GRAHPS")

try:
    with open('data/MyModel.pkl', 'rb') as file:
        print("Loading Model")
        model = pickle.load(file).to(device)
    with open('data/MyModelClass.pkl', 'rb') as file:
        classifier = pickle.load(file).to(device) 
        modelLoaded = True
except Exception: print("No model found. EXIT")

filenames = []
dataset = ss.search(#group = gr, 
                    #kind='Weighted', 
                    limit = 10000, 
                    rowbounds = (100,5000),
                    colbounds = (100,5000))
matrices = dataset.download(destpath=f'data/custom', extract=True)
for dataitem in dataset:
    filenames.append(dataitem.name)

names, graphs, lineGraphs, targets, stats = [], [], [], [], []

for filename in filenames:
    print("Current graph:", filename)
    names.append(filename)
    continue
    
    mmformat = mmread(f'data/custom/{filename}/{filename}.mtx').toarray()
    graph = dl.FromMMformat(mmformat)
    graphs.append(graph)
    lineGraphs.append(-1)
    
    if graph.edge_weight is None: graph.edge_weight = graph.edge_attr
    
    blossominput, target, uniqueEdges = [], [], set()
    for idx in range(len(graph.edge_index[0])):
        (i,j) = (graph.edge_index[0][idx].item(), graph.edge_index[1][idx].item())

        if (i,j) not in uniqueEdges and (j,i) not in uniqueEdges:
            blossominput.append((i, j, graph.edge_attr[idx].item()))        
            uniqueEdges.add((i,j))
            uniqueEdges.add((j,i))
    
    start_time = time.time()
    print("started blossom matching")
    match=maxWeightMatching(blossominput)
    timeOPT = time.time() - start_time
    print("done greedy matching")    
    target.append(match)
    targets.append(match)
    
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
        
    totalWeightOpt, duplicates = 0, set()
    for idx in true_edges_idx:
        
        from_node = graph.edge_index[0][idx].item()
        to_node = graph.edge_index[1][idx].item()
        
        if (from_node,to_node) in duplicates or (to_node,from_node) in duplicates: continue
        
        duplicates.add((from_node,to_node))
        duplicates.add((to_node,from_node))
        totalWeightOpt += graph.edge_weight[idx][0]
        
    print("done weight count")

    weightSum = 0
    print("Before reduction: ", graph.num_nodes)
    graph.x = lgc.AugmentOriginalNodeFeatures(graph)
    #graph, weightRed = rm.ApplyReductionRules(graph)
    #weightSum += weightRed
    print("After reduction: ", graph.num_nodes)
    start_time = time.time()
    print("started greedy matching")
    weightGreedy, pickedEdgeIndeces = gp.GreedyMatchingOrig(graph)
    print("done greedy matching")
    timeGreed = time.time() - start_time
    
    print("started gnn matching")
    graph = graph.to(device)
    start_time = time.time()
    graph, weightGnn = MyGCN.GNNMatching(model, classifier, graph.to(device))
    weightRes, pickedEdgeIndeces = gp.GreedyMatchingOrig(graph)
    weightSum += weightGnn
    weightSum += weightRes
    weightResDif = weightRes/weightSum 
    timeGNN = time.time() - start_time
    print("done gnn matching")

    stats.append([
        ("GNN: ", "TIME=", timeGNN, "W=", weightSum.item(), "GreedRES=", weightResDif.item()),
        ("GREED: ", "TIME=", timeGreed, "W=", weightGreedy, "", 0),
        ("OPT: ", "TIME=", timeOPT, "W=", totalWeightOpt.item(), "", 0)])
    

print("TOTAL GRAPHS FOUND: ", len(names))
# for i, g in enumerate(graphs):
#     print("-----------------------------")
    
#     print(names[i])
#     print("TOTAL NODES/EDGES:" , g.num_nodes, g.num_edges/2)
#     for s in stats[i]:
#         print(s)
#     print("-----------------------------")
    
file_name = 'data/custom/customdatasetfiles.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(names, file)
    print(f'Object successfully saved to "{file_name}"')    
    
    