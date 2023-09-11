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


m = torch.nn.LogSoftmax(dim=1)
t = torch.randn(2, 3)
output = m(t)
print(output)
# print("TEST NODE DELETION")
# original, converted_data, target = LoadData(1)

# print("BEFORE", converted_data)

# converted_data = ReduceGraph(converted_data[0], [5,9])

# print("AFTER", converted_data)