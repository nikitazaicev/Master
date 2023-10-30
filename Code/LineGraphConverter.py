import torch
import torch_geometric.utils as u
import networkx as nx

def PrintInfo(_data):
    print(_data)
    print(f'Number of nodes: {_data.num_nodes}')
    print(f'Number of edges: {_data.num_edges}')
    print(f'edges: {_data.edge_index}')

def CountDegree(graph):
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes
    deg = [0] * num_nodes
    for i in range(len(edge_index[0])):
        from_node = edge_index[0][i].item()
        #to_node = edge_index[1][i].item()
        deg[from_node]+=1
    return deg

def ToLineGraph(graph, edge_weight, verbose = False):

    num_edges = graph.num_edges
    edgeNodes = [set() for i in range(graph.num_nodes)]
    
    for i in range(len(graph.edge_index[0])):
        from_node = graph.edge_index[0][i]
        to_node = graph.edge_index[1][i]
        edgeNodes[from_node].add(i)
        edgeNodes[to_node].add(i)
    
    new_fromEdges = [] 
    new_toEdges = []    
    
    for i in range(len(graph.edge_index[0])):
        from_node = graph.edge_index[0][i]
        to_node = graph.edge_index[1][i]
        
        temp_edges = set()
        for n in edgeNodes[from_node]:
            temp_edges.add(n)
        for n in edgeNodes[to_node]:
            temp_edges.add(n)
        for n in temp_edges:
            if(i==n): continue
            new_fromEdges.append(i)
            new_toEdges.append(n)
    
    newEdgeIndex = torch.ones([2,len(new_fromEdges)], dtype=torch.int64)
    newEdgeIndex[0] = torch.tensor(new_fromEdges, dtype=torch.int64)
    newEdgeIndex[1] = torch.tensor(new_toEdges, dtype=torch.int64)
    graph.edge_index = newEdgeIndex
    graph.node_features = edge_weight[:]
    
    new_edgeWeights = torch.ones([len(graph.edge_index[0]),1])
    graph.edge_weight = new_edgeWeights.flatten()
    graph.edge_attr = new_edgeWeights.flatten()
    
    
    graph.x = graph.node_features.resize(len(graph.node_features),1)
    graph.pos = None
    graph.num_nodes = len(graph.node_features) 
    graph.num_edges = len(graph.edge_attr)
    
    degs = torch.Tensor(CountDegree(graph))
    degs = degs.unsqueeze(1)
    graph.x = torch.cat((graph.x, degs), dim=-1)
    
    assert(len(graph.edge_index[0])==len(graph.edge_weight))
    assert(graph.num_nodes==len(graph.x)) 
    
    assert(graph.num_nodes==num_edges)
    
    return graph
                
        
