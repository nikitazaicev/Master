import torch
import torch_geometric.utils as u
from MyDataLoader import Data
import numpy as np
import LineGraphConverter as lgc
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes
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
    weightSum, wasReduced, pickedNodeIds = torch.tensor([0.0]).to(device), True, set()
    pickedEdgeIds = set()
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
                pickedEdgeIds.add(i)
                wasReduced = True
        
    #rule 2 ?
    graph.adj = GenerateAdjListNoIdx(graph)
    return ReduceGraphOriginal(graph, pickedNodeIds), weightSum.item(), pickedEdgeIds

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

def ReduceGraph(graph, pickedNodeIds):
    #print("Removeing picked nodes and their neighbors from line graph", pickedNodeIds)
    print(graph)
    print(len(pickedNodeIds))
    subset = [x for x in range(graph.num_nodes) if x not in pickedNodeIds]
    new_edge_index, _ = u.subgraph(subset, 
                                   graph.edge_index,
                                   relabel_nodes = True)    
    new_node_feature = graph.node_features[subset]
    graph.edge_index = new_edge_index
    graph.node_features = new_node_feature
    
    new_degs = graph.x[subset]
    #new_degs = new_degs[:, [1,2,3,4]]
    graph.x = torch.reshape(new_node_feature, (-1,1))
    #graph.x = torch.cat((graph.x, new_degs), dim=-1)
    
    graph.num_nodes = len(graph.node_features)
    graph.num_edges = len(graph.edge_index[0])
    print(graph)
    #exit()
    return graph

def index_to_mask(index: Tensor, size: Optional[int] = None) -> Tensor:
    r"""Converts indices to a mask representation.

    Args:
        idx (Tensor): The indices.
        size (int, optional). The size of the mask. If set to :obj:`None`, a
            minimal sized output mask is returned.

    Example:

        >>> index = torch.tensor([1, 3, 5])
        >>> index_to_mask(index)
        tensor([False,  True, False,  True, False,  True])

        >>> index_to_mask(index, size=7)
        tensor([False,  True, False,  True, False,  True, False])
    """
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask
def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        if is_torch_sparse_tensor(edge_index):
            return max(edge_index.size(0), edge_index.size(1))
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))
def is_torch_sparse_tensor(src):
    r"""Returns :obj:`True` if the input :obj:`src` is a
    :class:`torch.sparse.Tensor` (in any sparse layout).

    Args:
        src (Any): The input object to be checked.
    """
    if isinstance(src, Tensor):
        if src.layout == torch.sparse_coo:
            return True
        if src.layout == torch.sparse_csr:
            return True
        if src.layout == torch.sparse_csc:
            return True
    return False    

def ReduceGraphOriginal(originalG, pickedNodeIds):  
    reducedOrig = Data()
    reducedOrig.x = lgc.UpdateNodeFeatures(originalG, originalG.adj, pickedNodeIds).to(device)
    subset = [x for x in range(originalG.num_nodes) if x not in pickedNodeIds]
    subset = torch.IntTensor(subset).to(device)
    
    new_edge_index, edge_weight = u.subgraph(subset, 
                                             originalG.edge_index, 
                                             originalG.edge_weight, 
                                             num_nodes=originalG.num_nodes, 
                                             relabel_nodes = True)    
    
    reducedOrig.edge_index = new_edge_index
    reducedOrig.edge_weight = edge_weight
    reducedOrig.edge_attr = edge_weight.flatten()
    reducedOrig.num_nodes = originalG.num_nodes-len(pickedNodeIds)
    reducedOrig.num_edges = len(reducedOrig.edge_index[0])
    reducedOrig.x = reducedOrig.x[subset]
    reducedOrig.adj = GenerateAdjListNoIdx(reducedOrig)
    
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

def GenerateAdjListNoIdx(graph):
    adjL = [set() for _ in range(graph.num_nodes)]
    for i in range(len(graph.edge_index[0])):
        w = graph.edge_weight[i].item()
        from_node = graph.edge_index[0][i].item()
        to_node = graph.edge_index[1][i].item()
        adjL[from_node].add((to_node,w))
        adjL[to_node].add((from_node,w))
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

