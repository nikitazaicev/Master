import torch
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import SuiteSparseMatrixCollection


def PrintInfo(_data):
    print(_data)
    print(f'Number of nodes: {_data.num_nodes}')
    print(f'Number of edges: {_data.num_edges}')
    print(f'edge_weight: {_data.edge_weight}')
    print(f'node_features: {_data.node_features}')
    print(f'edges: {_data.edge_index}')

def ToLineGraph(graph, verbose = False):
    
    print("-------- BEFORE --------")
    PrintInfo(graph)
    
    new_edgeWeights = torch.ones([len(graph.node_features),1], dtype=torch.float64)
    new_edge_index = torch.zeros([2, len(graph.node_features)], dtype=torch.float64)
    
    graph.node_features = graph.edge_weight[:]
    graph.edge_index = new_edge_index
    
    graph.edge_weight = new_edgeWeights
    graph.edge_attr = new_edgeWeights.flatten()
    graph.num_nodes = len(graph.node_features) 
    
    print("-------- AFTER --------")
    PrintInfo(graph)

    return graph

def FromLineGraph(graph):
    
    return graph


dataset = SuiteSparseMatrixCollection('/Data', 'Newman', 'netscience',transform=NormalizeFeatures())

train_data = dataset[0]
train_data.edge_weight=train_data.edge_attr.unsqueeze(1)
train_data.node_features=torch.ones(train_data.num_nodes,1)


train_data_converted = ToLineGraph(train_data, True)




