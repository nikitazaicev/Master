import torch
import torch.nn.functional as F
from torch_geometric.datasets import SuiteSparseMatrixCollection, GNNBenchmarkDataset
from ssgetpy import search, fetch
from DataLoader import LoadData
import numpy as np

np.random.seed(123)
torch.manual_seed(123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.version.cuda)

# data = fetch(group = 'HB', nzbounds = (None, 1000))
# print(data[0].edge_index)

def ReduceGraph(graph, nodes_to_remove, original = None):
    
    temp_edge_index = graph.edge_index
    temp_edge_weight = graph.edge_index
    temp_edge_attr = graph.edge_index
    
    for n in nodes_to_remove:
        for idx, from_node in enumerate(graph.edge_index[0]):
            if from_node == n: 
                from_nodes = graph.edge_index[0]
                to_nodes = graph.edge_index[1]
                from_nodes = torch.cat([graph.edge_index[0][:idx], graph.edge_index[0][idx+1:]])
                to_nodes = torch.cat([graph.edge_index[1][:idx], graph.edge_index[1][idx+1:]])
                temp = torch.cat([from_nodes,to_nodes])
                graph.edge_index = temp.view(2,len(graph.edge_index[0])-1)
                graph.edge_weight = torch.cat([graph.edge_weight[:idx], graph.edge_weight[idx+1:]])
                graph.edge_attr = torch.cat([graph.edge_attr[:idx], graph.edge_attr[idx+1:]])
                
        graph.node_features = torch.cat([graph.node_features[:n], graph.node_features[n+1:]])
        graph.x = torch.cat([graph.x[:n], graph.x[n+1:]])
    
    graph.num_nodes = len(graph.node_features) 
    graph.num_edges = len(graph.edge_attr) 
    
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
        
# original, converted_dataset, target = LoadData(5)    
# graph = converted_dataset[0]
# print(GenerateAdjMatrix(graph, graph.edge_weight)) 

out = torch.tensor([[-0.7230, -0.6642],
                    [-0.6654, -0.7217]])
print(torch.exp(out))
y = torch.tensor([1,0])
loss = F.nll_loss(out, y)
print(loss)









