import torch
import torch.nn.functional as F
from torch_geometric.datasets import SuiteSparseMatrixCollection, GNNBenchmarkDataset, KarateClub
from torch_geometric.transforms import NormalizeFeatures
from ssgetpy import search, fetch
from DataLoader import LoadData
import numpy as np

np.random.seed(123)
torch.manual_seed(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.version.cuda)

# data = fetch(group = 'HB', nzbounds = (None, 1000))
# print(data[0].edge_index)

def ReduceGraph(graph, pickedNodeIds):

    ids_toRemove = set()
    for n in pickedNodeIds:
        for idx, from_node in enumerate(graph.edge_index[0]):
            if from_node == n or graph.edge_index[1][idx] == n and idx not in ids_toRemove: 
                ids_toRemove.add(idx)

        graph.node_features[n] = 0
        graph.x[n][0] = 0

    edges_to_keep = len(graph.edge_index[0]) - len(ids_toRemove)
    new_edges = torch.zeros([2,edges_to_keep],dtype=torch.int)
    new_weights = torch.zeros([edges_to_keep],dtype=torch.float)
    new_atrs = torch.zeros([edges_to_keep],dtype=torch.float)     
    
    i = 0
    for idx in range(len(graph.edge_index[0])):    
        if idx in ids_toRemove: continue
        new_edges[0][i] = graph.edge_index[0][idx]
        new_edges[1][i] = graph.edge_index[1][idx]
        new_weights[i] = graph.edge_weight[idx]
        new_atrs[i] = graph.edge_attr[idx]
        i+=1
            
    graph.edge_index = new_edges
    graph.edge_weight = new_weights
    graph.edge_attr = new_atrs               
    
    #graph.num_nodes = len(graph.node_features) 
    graph.num_edges = len(graph.edge_attr) 

    assert(graph.num_nodes==graph.num_nodes)
    return graph

def GenerateAdjMatrix(graph, edge_weight):
    adjMat = torch.zeros(graph.num_nodes,graph.num_nodes)
    print(adjMat.size())
    for i in range(len(edge_weight)):
        from_node = graph.edge_index[0][i].item()
        to_node = graph.edge_index[1][i].item()
        adjMat[from_node][to_node] = edge_weight[i]
        adjMat[to_node][from_node] = edge_weight[i]
    return adjMat





