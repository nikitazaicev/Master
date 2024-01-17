import torch
import torch_geometric.utils as u
import GreedyPicker as gp
from blossom import maxWeightMatching
import DataLoader as dl
import torch.nn.functional as F
from torch_geometric.datasets import SuiteSparseMatrixCollection, GNNBenchmarkDataset, KarateClub, TUDataset
from torch_geometric.transforms import NormalizeFeatures, RemoveIsolatedNodes
from ssgetpy import search, fetch
from DataLoader import LoadData, Data
import numpy as np

np.random.seed(123)
torch.manual_seed(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.version.cuda)

# data = fetch(group = 'HB', nzbounds = (None, 1000))
# print(data[0].edge_index)


def ApplyReductionRules(graph):
    secondBestWs = [(0,0,-1,-1)] * graph.num_nodes # (max,max2,id,id2)
    adj = WeightedAdjList(graph)
    
    for i in range(len(graph.edge_index[0])):
        from_node = graph.edge_index[0][i]
        to_node = graph.edge_index[1][i]
        w = graph.edge_weight[i]
        
        n = from_node
        best, idx = secondBestWs[n][0], secondBestWs[n][2]
        if (best < w): secondBestWs[n] = (w,best,i,idx)
        best2, idx2 = secondBestWs[n][1], secondBestWs[n][3]
        if (best > w and best2 < w): 
            secondBestWs[n] = (secondBestWs[n][0],w,secondBestWs[n][2],i)
        
        n = to_node
        best, idx = secondBestWs[n][0], secondBestWs[n][2]
        if (best < w): secondBestWs[n] = (w,best,i,idx)
        best2, idx2 = secondBestWs[n][1], secondBestWs[n][3]
        if (best > w and best2 < w): 
            secondBestWs[n] = (secondBestWs[n][0],w,secondBestWs[n][2],i)
       
    #rule 1 dominating edge
    weightSum, wasReduced, pickedNodeIds = torch.tensor([0.0]), True, set()
    while wasReduced and len(pickedNodeIds) < graph.num_nodes:
        wasReduced = False
        for i in range(len(graph.edge_index[0])):
            from_node, to_node = graph.edge_index[0][i].item(), graph.edge_index[1][i].item()
            if from_node in pickedNodeIds or to_node in pickedNodeIds: continue
            w = graph.edge_weight[i]
            
            FromNodeBest, FromNodeBest2 = secondBestWs[from_node][0], secondBestWs[from_node][1]
            FromNodeBestId, FromNodeBest2Id = secondBestWs[from_node][2], secondBestWs[from_node][3]
            ToNodeBest, ToNodeBest2 = secondBestWs[to_node][0], secondBestWs[to_node][1]
            ToNodeBestId, ToNodeBest2Id = secondBestWs[to_node][2], secondBestWs[to_node][3]
            
            neighborSum = 0
            if FromNodeBestId != i: neighborSum += FromNodeBest 
            else: neighborSum += FromNodeBest2
            if ToNodeBestId != i: neighborSum += ToNodeBest
            else: neighborSum += ToNodeBest2
            
            if w >= neighborSum: 

                ReduceBestWeightsTable(graph, secondBestWs, i, adj)
                weightSum += w
                
                pickedNodeIds.add(from_node)
                pickedNodeIds.add(to_node)
                wasReduced = True
        
    #rule 2 ?

    return ReduceGraphOriginal(graph, pickedNodeIds), weightSum.item()

def ReduceBestWeightsTable(graph, secondBestWs, deleteEdge, adj):
    pickedNodes = {graph.edge_index[0][deleteEdge].item(), graph.edge_index[1][deleteEdge].item()}
    for n in pickedNodes:
        for (neighbor,w,eId) in adj[n]:
            if neighbor in pickedNodes: continue
            bestW, bestW2 = secondBestWs[neighbor][0], secondBestWs[neighbor][1]
            bestId, best2Id = secondBestWs[neighbor][2], secondBestWs[neighbor][3]
            if bestId == eId or best2Id == eId: 
                nextBest, nextBestI = 0, -1
                for (idx,weight,edgeId) in adj[neighbor]:
                    if (bestId != edgeId and best2Id != edgeId) and weight > nextBest: 
                        nextBest = w 
                        nextBestI = idx
                        
                if bestId == eId:
                    secondBestWs[neighbor] = (bestW2,nextBest,best2Id,nextBestI)
                if best2Id == eId:
                    secondBestWs[neighbor] = (bestW,nextBest,bestId,nextBestI)
                
            adj[neighbor].remove((n,w,eId))

    return

def ReduceGraph(original, graph, pickedNodeIds):
    #print("Removeing picked nodes and their neighbors from line graph", pickedNodeIds)

    subset = [x for x in range(graph.num_nodes) if x not in pickedNodeIds]
    new_edge_index, _ = u.subgraph(subset, graph.edge_index, relabel_nodes = True)    
    new_node_feature = graph.node_features[subset]
    graph.edge_index = new_edge_index
    graph.node_features = new_node_feature
    
    new_degs = graph.x[subset]
    new_degs = new_degs[:, [1,2,3,4]]
    graph.x = torch.reshape(new_node_feature, (-1,1))
    graph.x = torch.cat((graph.x, new_degs), dim=-1)
    
    graph.num_nodes = len(graph.node_features)
    graph.num_edges = len(graph.edge_index[0])
    
    #print("Removeing picked edges and their neighbors from original graph", pickedNodeIds)
    reducedOrig = Data()
    reducedOrig.num_nodes = original.num_nodes
    reducedOrig.x = original.x
    reducedOrig.edge_index = torch.zeros([2,len(original.edge_index[0])-len(pickedNodeIds)])
    reducedOrig.edge_weight = torch.zeros([len(original.edge_index[0])-len(pickedNodeIds)])
    reducedOrig.edge_attr = torch.zeros([len(original.edge_index[0])-len(pickedNodeIds)])
    counter = 0
    for i in range(len(original.edge_index[0])):
        if i not in pickedNodeIds:
            reducedOrig.edge_index[0][counter] = original.edge_index[0][i]
            reducedOrig.edge_index[1][counter] = original.edge_index[1][i]
            reducedOrig.edge_weight[counter] = original.edge_weight[i]
            reducedOrig.edge_attr[counter] = original.edge_attr[i]
            counter += 1
            
    reducedOrig.num_edges = len(reducedOrig.edge_index[0])
    assert((len(reducedOrig.edge_index[0])==graph.num_nodes))
    return reducedOrig, graph

def ReduceGraphOriginal(originalG, pickedNodeIds):  
    reducedOrig = Data()
    subset = [x for x in range(originalG.num_nodes) if x not in pickedNodeIds]
    new_edge_index, edge_weight = u.subgraph(subset, originalG.edge_index, originalG.edge_weight, relabel_nodes = True)    
    new_node_feature = originalG.node_features[subset]
    reducedOrig.edge_index = new_edge_index
    reducedOrig.edge_weight = edge_weight 
    reducedOrig.node_features = new_node_feature
    
    reducedOrig.x = originalG.x[subset]
    
    reducedOrig.num_nodes = len(reducedOrig.x)
    reducedOrig.num_edges = len(reducedOrig.edge_index[0])

    assert(len(reducedOrig.edge_weight)==len(reducedOrig.edge_index[0]))
    assert((len(reducedOrig.edge_index[0])==reducedOrig.num_edges))
    assert((len(reducedOrig.x)==reducedOrig.num_nodes))
    assert((originalG.num_nodes-len(pickedNodeIds))==reducedOrig.num_nodes)
    return reducedOrig

def GenerateAdjMatrix(graph):
    adjMat = torch.zeros(graph.num_nodes,graph.num_nodes)
    edge_weight = graph.edge_weight
    for i in range(len(edge_weight)):
        from_node = graph.edge_index[0][i].item()
        to_node = graph.edge_index[1][i].item()
        adjMat[from_node][to_node] = edge_weight[i]
        adjMat[to_node][from_node] = edge_weight[i]
    return adjMat

def GenerateAdjList(graph):
    adjL = [set() for _ in range(graph.num_nodes)]
    for i in range(len(graph.edge_index[0])):
        from_node = graph.edge_index[0][i].item()
        to_node = graph.edge_index[1][i].item()
        adjL[from_node].add(to_node)
        adjL[to_node].add(from_node)
    return adjL

def WeightedAdjList(graph):
    adjL = [set() for _ in range(graph.num_nodes)]
    for i in range(len(graph.edge_index[0])):
        w = graph.edge_weight[i].item()
        from_node = graph.edge_index[0][i].item()
        to_node = graph.edge_index[1][i].item()
        adjL[from_node].add((to_node,w,i))
        adjL[to_node].add((from_node,w,i))
    return adjL

