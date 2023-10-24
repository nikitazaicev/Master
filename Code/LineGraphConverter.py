import torch
import torch_geometric.utils as u
import networkx as nx

def PrintInfo(_data):
    print(_data)
    print(f'Number of nodes: {_data.num_nodes}')
    print(f'Number of edges: {_data.num_edges}')
    print(f'edges: {_data.edge_index}')


def ToLineGraph(graph, edge_weight, verbose = False):
    # print("---------------------------------------------------------")    
    # print(graph.edge_index[0])
    # print(graph.edge_index[1])
    # print("---------------------------------------------------------")
    
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
    
    
    # print("---------------------------------------------------------")    
    # print(graph.num_nodes, num_edges)
    # print(graph.edge_index[0])
    # print(graph.edge_index[1])
    # print("---------------------------------------------------------")
    
    assert(len(graph.edge_index[0])==len(graph.edge_weight))
    assert(graph.num_nodes==len(graph.x)) 
    
    assert(graph.num_nodes==num_edges)
    # exit()
    
   
    
    return graph
        

def ToLineGraph2(graph, edge_weight, verbose = False):

    G = u.to_networkx(graph,
                    node_attrs=['x'],
                    edge_attrs=['edge_attr'],
                    to_undirected=True)
    line_graph = nx.line_graph(G, create_using=nx.DiGraph)
    res_data = u.from_networkx(line_graph)
    
    graph.edge_index = res_data.edge_index
    graph.node_features = edge_weight[:]
    
    new_edgeWeights = torch.ones([len(res_data.edge_index[0]),1])
    graph.edge_weight = new_edgeWeights.flatten()
    graph.edge_attr = new_edgeWeights.flatten()
    
    graph.x = graph.node_features.resize(len(graph.node_features),1)
    graph.pos = None
    graph.num_nodes = len(graph.node_features) 
    graph.num_edges = len(graph.edge_attr) 
    
    return graph
        
        
