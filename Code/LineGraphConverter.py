import torch
import torch_geometric.utils as u
import torch.nn.functional as F

def PrintInfo(_data):
    print(_data)
    print(f'Number of nodes: {_data.num_nodes}')
    print(f'Number of edges: {_data.num_edges}')
    print(f'edges: {_data.edge_index}')


def GenerateAdjList(graph):
    adjL = [set() for _ in range(graph.num_nodes)]
    for i in range(len(graph.edge_index[0])):
        from_node = graph.edge_index[0][i].item()
        to_node = graph.edge_index[1][i].item()
        adjL[from_node].add(to_node)
        adjL[to_node].add(from_node)
    return adjL

def CountExtraFeatures(graph):
    num_nodes = graph.num_nodes
    rels = [0.0] * num_nodes
    diffs = [0.0] * num_nodes
    maxs = [0.0] * num_nodes
    adj = GenerateAdjList(graph)
    for i, neighbors in enumerate(adj):
        w = graph.node_features[i]
        neighborW = 0
        for n in neighbors:
            neighborW = neighborW + graph.node_features[n]
            diffs[i] = diffs[i] + (w - graph.node_features[n])
            maxs[i] = maxs[i] + graph.node_features[n]
        rels[i] = w/neighborW
    rels = torch.Tensor(rels).unsqueeze(1)
    diffs = torch.Tensor(diffs).unsqueeze(1)
    maxs = torch.Tensor(maxs).unsqueeze(1)
    return torch.cat((rels, diffs, maxs), dim=-1)

def CountDegree(graph):
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes
    deg = [0] * num_nodes
    for i in range(len(edge_index[0])):
        from_node = edge_index[0][i].item()
        #to_node = edge_index[1][i].item()
        deg[from_node]+=1
    return torch.Tensor(deg).unsqueeze(1)

def AugmentNodeFeatures(graph, isLIneGraph=False):
    
    if isLIneGraph:
        graph.x = graph.node_features.resize(len(graph.node_features),1)
    else:
        graph.x = torch.ones([graph.num_nodes,1])
        graph.node_features = graph.x
    
    degs = CountDegree(graph)
    graph.x = torch.cat((graph.x, degs), dim=-1)
    feats = CountExtraFeatures(graph)
    graph.x = torch.cat((graph.x, feats), dim=-1)
    if isLIneGraph: assert(graph.x[0][0]==1)
    return graph.x

def ToLineGraph(graph, edge_weight, verbose = False):

    num_edges = graph.num_edges
    edgeNodes = [set() for i in range(graph.num_nodes)]
    
    for i in range(len(graph.edge_index[0])):
        from_node = graph.edge_index[0][i].item()
        to_node = graph.edge_index[1][i].item()
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
            if(i==n): continue
            new_fromEdges.append(i)
            new_toEdges.append(n)
        
        for n in edgeNodes[to_node]:
            temp_edges.add(n)
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
    
    graph.pos = None
    graph.num_nodes = len(graph.node_features) 
    graph.num_edges = len(graph.edge_attr)
    
    AugmentNodeFeatures(graph)
    
    #assert(len(graph.x[0])==1)
    assert(len(graph.edge_index[0])==len(graph.edge_weight))
    assert(graph.num_nodes==len(graph.x)) 
    
    assert(graph.num_nodes==num_edges)
    
    return graph
                
        
