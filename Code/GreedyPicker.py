import torch
import ReductionsManager as rm
from torch_geometric.datasets import SuiteSparseMatrixCollection, GNNBenchmarkDataset, KarateClub, TUDataset
from torch_geometric.transforms import NormalizeFeatures, RemoveIsolatedNodes
from DataLoader import RemoveDoubleEdges, FromMMformat
import pickle
from blossom import maxWeightMatching
import ssgetpy as ss
from scipy.io import mmread


def GreedyMatchingOrig(graph, prevEdgeIndeces=None, verbose=False):
    
    totalWeight = 0
    if prevEdgeIndeces is None: pickedEdgeIndeces = set()
    else: pickedEdgeIndeces = prevEdgeIndeces
    takenNodes = set()
    edge_weights = graph.edge_weight.flatten()

    sorted_edges = torch.sort(edge_weights, descending=True)
    edge_index = graph.edge_index
    for i, original_index in enumerate(sorted_edges.indices):
        original_index = original_index.item()
        from_node = edge_index[0][original_index].item()
        to_node = edge_index[1][original_index].item()
        if (from_node not in takenNodes 
            and to_node not in takenNodes
            and original_index not in pickedEdgeIndeces):
            takenNodes.add(from_node)
            takenNodes.add(to_node)
            pickedEdgeIndeces.add(original_index)
            totalWeight += edge_weights[original_index].item()

            
    if verbose: print(f"Greedy matching total weight result = {totalWeight:.4f}")
    if verbose: print(f"In total {len(pickedEdgeIndeces)} out of {len(edge_weights)} edges were picked")
    if verbose: print(f"{len(takenNodes)} out of {graph.num_nodes} nodes in the matching")
    return totalWeight, pickedEdgeIndeces 


def GreedyMatchingLine(graph):
    
    totalWeight = 0

    pickedEdgeIndeces = set()
    sorted_edges = torch.sort(graph.x[:, [0]].flatten(), descending=True)    
    
    adj = rm.GenerateAdjList(graph)
    count = 0
    for i, original_index in enumerate(sorted_edges.indices):
        original_index = original_index.item()
        
        if (original_index not in pickedEdgeIndeces):
            pickedEdgeIndeces.add(original_index)
            totalWeight += graph.x[original_index][0].item()
            pickedEdgeIndeces.update(adj[original_index])
            count += 1

    print(f"Greedy matching total weight result = {totalWeight:.4f}")
    print(f"In total {count} out of {len(sorted_edges.indices)} edges were picked")
    return totalWeight, pickedEdgeIndeces 

def RandomMatching(graph):
    
    totalWeight = 0.0
    pickedEdgeIndeces = []
    takenNodes = set()
    
    edge_weights = graph.edge_weight
    perm = torch.randperm(len(graph.edge_index[0]))
    edge_index = graph.edge_index
    
    for i in perm:
        from_node = edge_index[0][i].item()
        to_node = edge_index[1][i].item()
        if from_node not in takenNodes and to_node not in takenNodes:
            takenNodes.add(from_node)
            takenNodes.add(to_node)
            pickedEdgeIndeces.append(i)            
            totalWeight += edge_weights[i].item()
                 
    print(f"Random matching total weight result = {totalWeight:.4f}")
    print(f"In total {len(pickedEdgeIndeces)} out of {len(edge_weights)} edges were picked")
    print(f"{len(takenNodes)} out of {graph.num_nodes} nodes in the matching")
    return totalWeight, pickedEdgeIndeces 

def GreedyScores(pred, graph, original_g, threshold = 0.5):
    picked_nodes = set()
    picked_edgeIds = set()
    weightSum = 0
    sorted_pred = torch.sort(pred, descending=True)

    for i, sorted_i in enumerate(sorted_pred.indices):
        sorted_i = sorted_i
        from_node = original_g.edge_index[0][sorted_i].item()
        to_node = original_g.edge_index[1][sorted_i].item()
        
        if (sorted_pred.values[i] >= threshold 
            and (from_node not in picked_nodes 
            and to_node not in picked_nodes)):
            weightSum += original_g.edge_attr[sorted_i]
            picked_edgeIds.add(sorted_i.item())
            picked_nodes.add(from_node)
            picked_nodes.add(to_node)
    picked_neighbors = set()
    for i in range(len(graph.edge_index[0])):
        from_node = graph.edge_index[0][i].item()
        to_node = graph.edge_index[1][i].item()
        if from_node in picked_edgeIds:
            picked_neighbors.add(to_node)
        if to_node in picked_edgeIds:
            picked_neighbors.add(from_node)
    
    excluded_edgeIds = set()    
    excluded_edgeIds.update(picked_edgeIds)                
    excluded_edgeIds.update(picked_neighbors)
    return weightSum, excluded_edgeIds, len(picked_edgeIds)

def GreedyScoresOriginal(pred, original_g, threshold = 0.5):
    picked_nodes = set()
    picked_edgeIds = set()
    weightSum = 0
    sorted_pred = torch.sort(pred, descending=True)

    picked_edges = set()
    for i, sorted_i in enumerate(sorted_pred.indices):
        from_node = original_g.edge_index[0][sorted_i].item()
        to_node = original_g.edge_index[1][sorted_i].item()
        
        if (sorted_pred.values[i] >= threshold 
            and (from_node not in picked_nodes 
            and to_node not in picked_nodes)):
            weightSum += original_g.edge_weight[sorted_i][0]
            picked_edgeIds.add(sorted_i.item())
            picked_nodes.add(from_node)
            picked_nodes.add(to_node)
            picked_edges.add(sorted_i)
    
    return weightSum, picked_edges, picked_nodes

# DATASET CANDIDATES
# webbase-2001
# it-2004
# GAP-twitter 
# twitter7
# GAP-web
# sk-2005

# gr = 'Newman'
# dataset = ss.search(group = gr, limit = 100, rowbounds = (10,10000), colbounds = (10,10000) )
# filenames = []
# for dataitem in dataset:
#     filenames.append(dataitem.name)

# matrices = dataset.download(destpath=f'data/{gr}', extract=True)
# dataset = []
# converted_dataset = []
# for filename in filenames:
#     mmformat = mmread(f'data/{gr}/{filename}/{filename}.mtx').toarray()
#     original_graph = FromMMformat(mmformat)
#     #copy_graph = FromMMformat(mmformat)
#     print("START...")
#     #print("LINE GRAPH PROCCESSING...")
#     #print(copy_graph.num_nodes)
#     #if copy_graph.num_nodes == 0 or copy_graph.num_nodes == 487: continue
#     #edgesbefore = copy_graph.num_edges
#     #line_graph = ToLineGraph(copy_graph, copy_graph.edge_attr, verbose = False)
#     #print(copy_graph.num_edges)
#     #assert(copy_graph.num_nodes==edgesbefore)
#     #assert(copy_graph.num_edges<=(0.5*(edgesbefore)*(edgesbefore-1))+1)
    
#     print("EDGES: ", len(original_graph.edge_index[0]))
#     print("EDGE ATTRS : ", original_graph.edge_attr)
#     if original_graph.edge_weight is None: original_graph.edge_weight = original_graph.edge_attr
#     print("EDGE WEIGHTS : ", original_graph.edge_weight)
    
    
#     totalWeightGreed, pickedEdgeIndeces = GreedyMatchingOrig(original_graph)
#     blossominput = []
#     target = []
#     for i in range(len(original_graph.edge_index[0])):
#         blossominput.append((original_graph.edge_index[0][i].item(),
#                               original_graph.edge_index[1][i].item(),
#                               original_graph.edge_attr[i].item()))
    
#     match=maxWeightMatching(blossominput)
#     target.append(match)
    
#     val_y = target
#     val_y_item = torch.LongTensor(val_y[0])
#     classes = []
#     for j in range(len(original_graph.edge_index[0])):
#         from_node = original_graph.edge_index[0][j].item()
#         to_node = original_graph.edge_index[1][j].item() 
#         if val_y_item[from_node] == to_node:
#             classes.append(1)
#         else:
#             classes.append(0)
#     val_y_item = torch.FloatTensor(classes)
#     true_edges_idx = (val_y_item == 1).nonzero(as_tuple=True)[0]
#     totalWeightOpt = 0
#     for idx in true_edges_idx:
#         from_node = original_graph.edge_index[0][idx].item()
#         to_node = original_graph.edge_index[1][idx].item()
#         totalWeightOpt += original_graph.edge_attr[idx]
    
#     graphs = []
#     print("----------------------------------")
#     print("GREEDY WEIGHT = ", totalWeightGreed)
#     print("OPTIMAL WEIGHT = ", totalWeightOpt)
#     print("DIFF = ", totalWeightGreed/totalWeightOpt)
    

#     if (totalWeightGreed/totalWeightOpt).item() < 0.8:
#         try:
#             with open('data/OptGreedDiffDataPaths.pkl', 'rb') as file:
#                 matchCriteriaData = pickle.load(file)
#         except Exception: matchCriteriaData = dict()
            
#         print("FOUND SUITABLE DATA!!!")
#         print(original_graph)
#         matchCriteriaData[f'data/{gr}/{filename}/{filename}.mtx'] = original_graph
#         with open("data/OptGreedDiffDataPaths.pkl", 'wb') as file:
#             pickle.dump(matchCriteriaData, file)
    
    
#     print("DONE")
#     #converted_dataset.append(line_graph)
#     #dataset.append((original_graph,f'data/{gr}/{filename}/{filename}.mtx'))

    
# try:
#     with open('data/OptGreedDiffDataPaths.pkl', 'rb') as file:
#         matchCriteriaData = pickle.load(file)    
# except Exception: matchCriteriaData = dict()
    
# print("TOTAL DATA FOUND = ", len(matchCriteriaData))
# print("----------------------------------")
