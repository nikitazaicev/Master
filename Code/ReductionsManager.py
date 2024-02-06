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
    topWeights = [(0,0,0,-1,-1,-1,-1,-1,-1)] * graph.num_nodes # (max,max2,max3,id,id2,id3,endNode,endNode2,endNode3)
    adj = WeightedAdjList(graph)
    
    for i in range(len(graph.edge_index[0])):
        from_node = graph.edge_index[0][i]
        to_node = graph.edge_index[1][i]
        w = graph.edge_weight[i]
        
        n = from_node
        best, idx, endNode = topWeights[n][0], topWeights[n][3], topWeights[n][6]
        if (best < w): 
            topWeights[n] = (w,best,topWeights[n][2],
                               i,idx,topWeights[n][5],
                               to_node,endNode,topWeights[n][8])
        
        best2, idx2, endNode2 = topWeights[n][1], topWeights[n][4], topWeights[n][7]
        if (best > w and best2 < w): 
            topWeights[n] = (topWeights[n][0],w,best2,
                               topWeights[n][3],i,idx2,
                               topWeights[n][6],to_node,endNode2)
        
        best3, idx3, endNode3 = topWeights[n][2], topWeights[n][5], topWeights[n][8]
        if (best2 > w and best3 < w): 
            topWeights[n] = (topWeights[n][0],topWeights[n][1],w,
                               topWeights[n][3],topWeights[n][4],i,
                               topWeights[n][6],topWeights[n][7],to_node)
        
        n = to_node
        best, idx, endNode = topWeights[n][0], topWeights[n][3], topWeights[n][6]
        if (best < w): 
            topWeights[n] = (w,best,topWeights[n][2],
                               i,idx,topWeights[n][5],
                               to_node,endNode,topWeights[n][8])
        
        best2, idx2, endNode2 = topWeights[n][1], topWeights[n][4], topWeights[n][7]
        if (best > w and best2 < w): 
            topWeights[n] = (topWeights[n][0],w,best2,
                               topWeights[n][3],i,idx2,
                               topWeights[n][6],to_node,endNode2)
        
        best3, idx3, endNode3 = topWeights[n][2], topWeights[n][5], topWeights[n][8]
        if (best2 > w and best3 < w): 
            topWeights[n] = (topWeights[n][0],topWeights[n][1],w,
                               topWeights[n][3],topWeights[n][4],i,
                               topWeights[n][6],topWeights[n][7],to_node)
       
    #rule 1 dominating edge
    weightSum, wasReduced, pickedNodeIds = torch.tensor([0.0]), True, set()
    while wasReduced and len(pickedNodeIds) < graph.num_nodes:
        wasReduced = False
        for i in range(len(graph.edge_index[0])):
            from_node, to_node = graph.edge_index[0][i].item(), graph.edge_index[1][i].item()
            if from_node in pickedNodeIds or to_node in pickedNodeIds: continue
            w = graph.edge_weight[i]
            
            FromBest, FromBest2, FromBest3 = topWeights[from_node][0], topWeights[from_node][1], topWeights[from_node][2]
            FromBestId, FromBestId2, FromBestId3  = topWeights[from_node][3], topWeights[from_node][4], topWeights[from_node][5]
            FromBestEnd, FromBestEnd2, FromBestEnd3  = topWeights[from_node][6], topWeights[from_node][7], topWeights[from_node][8]
            
            ToBest, ToBest2, ToBest3 = topWeights[to_node][0], topWeights[to_node][1], topWeights[to_node][2]
            ToBestId, ToBestId2, ToBestId3  = topWeights[to_node][3], topWeights[to_node][4], topWeights[to_node][5]
            ToBestEnd, ToBestEnd2, ToBestEnd3  = topWeights[to_node][6], topWeights[to_node][7], topWeights[to_node][8]
            
            neighborSum = 0

            if FromBestId == i: 
                FromBest, FromBestId, FromBestEnd = FromBest2, FromBestId2, FromBestEnd2
                FromBest2, FromBestId2, FromBestEnd2 = FromBest3, FromBestId3, FromBestEnd3
            elif FromBestId2 == i: 
                FromBest2, FromBestId2, FromBestEnd2 = FromBest3, FromBestId3, FromBestEnd3

            if ToBestId == i: 
                ToBest, ToBestId, ToBestEnd = ToBest2, ToBestId2, ToBestEnd2
                ToBest2, ToBestId2, ToBestEnd2 = ToBest3, ToBestId3, ToBestEnd3
            elif ToBestId2 == i: 
                ToBest2, ToBestId2, ToBestEnd2 = ToBest3, ToBestId3, ToBestEnd3
            
            if FromBestEnd != ToBestEnd: 
                neighborSum += FromBest + ToBest 
            elif FromBest > ToBest: 
                neighborSum += FromBest + ToBest2
            else: neighborSum += FromBest2 + ToBest
                
            
            if w >= neighborSum: 
                
                # print((from_node, to_node), w, " >= ", neighborSum)
                # print(adj[from_node])
                # print(adj[to_node])
                # print(FromBest,FromBest2,FromBestEnd,FromBestEnd2)
                # print(ToBest,ToBest2,ToBestEnd,ToBestEnd2)
                # exit(1)
                ReduceBestWeightsTable(graph, topWeights, i, adj)
                weightSum += w
                
                pickedNodeIds.add(from_node)
                pickedNodeIds.add(to_node)
                wasReduced = True
        
    #rule 2 ?

    return ReduceGraphOriginal(graph, pickedNodeIds), weightSum.item()

def ReduceBestWeightsTable(graph, topWeights, deleteEdge, adj):
    pickedNodes = {graph.edge_index[0][deleteEdge].item(), graph.edge_index[1][deleteEdge].item()}
    for n in pickedNodes:
        for (neighbor,w,eId) in adj[n]:
            if neighbor in pickedNodes: continue
            bestW, bestW2, bestW3 = topWeights[neighbor][0], topWeights[neighbor][1], topWeights[neighbor][2]
            bestId, bestId2, bestId3 = topWeights[neighbor][3], topWeights[neighbor][4], topWeights[neighbor][5]
            nodeEnd, nodeEnd2, nodeEnd3 = topWeights[neighbor][6], topWeights[neighbor][7], topWeights[neighbor][8]
            if bestId == eId or bestId2 == eId or bestId3 == eId: 
                
                nextBest, nextBestId, nextEndNode = 0, -1, -1
                for (endNode,weight,edgeId) in adj[neighbor]:
                    if (bestId != edgeId and bestId2 != edgeId and bestId3 != edgeId) and weight > nextBest: 
                        nextBest = w 
                        nextBestId = edgeId
                        nextEndNode = endNode
                        
                if bestId == eId:
                    topWeights[neighbor] = (bestW2,bestW3,nextBest,
                                            bestId2,bestId3,nextBestId,
                                            nodeEnd2, nodeEnd3, nextEndNode)
                if bestId2 == eId:
                    topWeights[neighbor] = (bestW,bestW3,nextBest,
                                            bestId,bestId3,nextBestId,
                                            nodeEnd, nodeEnd3, nextEndNode)
                if bestId3 == eId:
                    topWeights[neighbor] = (bestW,bestW2,nextBest,
                                            bestId,bestId2,nextBestId,
                                            nodeEnd,nodeEnd2,nextEndNode)
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

