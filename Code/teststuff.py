import torch
import torch_geometric.utils as u
import GreedyPicker as gp
from blossom import maxWeightMatching
import DataLoader as dl
import torch.nn.functional as F
from torch_geometric.datasets import SuiteSparseMatrixCollection, GNNBenchmarkDataset, KarateClub, TUDataset
from torch_geometric.transforms import NormalizeFeatures, RemoveIsolatedNodes
from ssgetpy import search, fetch
from DataLoader import LoadData, LoadTestData, Data
import numpy as np

np.random.seed(123)
torch.manual_seed(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.version.cuda)

# data = fetch(group = 'HB', nzbounds = (None, 1000))
# print(data[0].edge_index)

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

