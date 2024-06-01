import torch

def PrintInfo(_data):
    print(_data)
    print(f'Number of nodes: {_data.num_nodes}')
    print(f'Number of edges: {_data.num_edges}')
    print(f'edges: {_data.edge_index}')


def GenerateAdjList(graph):
    adjL = [set() for _ in range(graph.num_nodes)]
    for i in range(len(graph.edge_index[0])):
        w = graph.edge_weight[i].item()
        from_node = graph.edge_index[0][i].item()
        to_node = graph.edge_index[1][i].item()
        adjL[from_node].add((to_node,w))
        adjL[to_node].add((from_node,w))
    return adjL

def CountExtraFeatures(graph, adj):
    num_nodes = graph.num_nodes
    rels = [0.0] * num_nodes
    diffs = [0.0] * num_nodes
    maxs = [0.0] * num_nodes
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

def CountExtraFeaturesOriginal(graph, adj):
    num_nodes = graph.num_nodes
    rels = [0.0] * num_nodes
    diffs = [0.0] * num_nodes
    sums = [0.0] * num_nodes
    #maxs = [0.0] * num_nodes
    #maxs2 = [0.0] * num_nodes
    
    for i in range(len(graph.edge_index[0])):
        from_node = graph.edge_index[0][i]
        to_node = graph.edge_index[1][i]
        w = graph.edge_weight[i]
        sums[from_node] = sums[from_node] + w
        sums[to_node] = sums[to_node] + w
        
        # best = maxs[from_node]
        # if best < w: 
        #     maxs2[from_node] = best
        #     maxs[from_node] = w 
        # best = maxs[to_node]
        # if best < w: 
        #     maxs2[to_node] = best
        #     maxs[to_node] = w 

    for i, neighbors in enumerate(adj):
        neighborSums = 0
        for (n,w) in neighbors:
            neighborSums = neighborSums + sums[n]
        diffs[i] = (sums[i] - neighborSums)
        if neighborSums > 0: rels[i] = sums[i]/neighborSums   
    
    rels = torch.Tensor(rels).unsqueeze(1)
    diffs = torch.Tensor(diffs).unsqueeze(1)
    sums = torch.Tensor(sums).unsqueeze(1)
    #maxs = torch.Tensor(maxs).unsqueeze(1)
    #maxs2 = torch.Tensor(maxs2).unsqueeze(1)
    
    return torch.cat((rels, diffs, sums), dim=-1) 
    #return torch.cat((rels, diffs, sums, maxs, maxs2), dim=-1)

def UpdateExtraFeatures(graph, adj, removedNodeIds):
    for nodeId in removedNodeIds:
        for (n,w) in adj[nodeId]:
            graph.x[n][4] -= w
            graph.x[n][3] = graph.x[n][3] - w + graph.x[nodeId][4]   
            neighborSums = 0
            for (n2,w2) in adj[n]:
                if n2 != nodeId: neighborSums += w2
            if neighborSums > 0: graph.x[n][2] = graph.x[n][4] / neighborSums    
            else: graph.x[n][2] = 0
            adj[n].remove((nodeId,w))
    return graph.x

def CountDegree(graph, adj, undirected = True):
    edge_index = graph.edge_index
    num_nodes = graph.num_nodes
    deg = [0] * num_nodes
    for i in range(len(edge_index[0])):
        from_node = edge_index[0][i].item()
        to_node = edge_index[1][i].item()
        deg[from_node]+=1
        if not undirected: deg[to_node]+=1
    return torch.Tensor(deg).unsqueeze(1)

def AugmentNodeFeatures(graph, adj, undirected = True):
    graph.x = graph.node_features.resize(len(graph.node_features),1)
    graph.node_features = graph.x 
    
    degs = CountDegree(graph, adj, undirected)
    graph.x = torch.cat((graph.x, degs), dim=-1)
    feats = CountExtraFeatures(graph, adj)
    graph.x = torch.cat((graph.x, feats), dim=-1)
    return graph.x

def AugmentOriginalNodeFeatures(graph, undirected = False):
    adj = GenerateAdjList(graph)
    graph.x = torch.ones([graph.num_nodes,1])
    graph.node_features = graph.x
    
    degs = CountDegree(graph, adj, undirected)

    graph.x = torch.cat((graph.x, degs), dim=-1)
    feats = CountExtraFeaturesOriginal(graph, adj)
    graph.x = torch.cat((graph.x, feats), dim=-1)
    return graph.x, adj 

def UpdateNodeFeatures(graph, adj, removedNodeIds):
    
    UpdateDegree(graph, adj, removedNodeIds)
    UpdateExtraFeatures(graph, adj, removedNodeIds)
    #graph.x, adj = AugmentOriginalNodeFeatures(graph)
    return graph.x

def UpdateDegree(graph, adj, removedNodeIds):
    for nodeId in removedNodeIds:
        for (neighbor,w) in adj[nodeId]:
            graph.x[neighbor][1] -= 1    
        
    return graph.x

def RemoveDoubleEdges(graph):
    unique = set()
    idx = dict()
    deleted = 0
    total_edges = len(graph.edge_index[0])

    for i in range(total_edges):
        from_node = graph.edge_index[0][i].item()
        to_node = graph.edge_index[1][i].item()
        if (from_node,to_node) in unique or (to_node, from_node) in unique:
            deleted += 1
            continue
        unique.add((from_node,to_node))
        idx[(from_node,to_node)] = i
    new_edges = torch.zeros([2,len(unique)],dtype=torch.int)
    new_weights = torch.zeros([len(unique)],dtype=torch.float)
    new_atrs = torch.zeros([len(unique)],dtype=torch.float)
    unique  = sorted(unique)
    for i, u  in enumerate(unique):
        
        new_edges[0][i] = torch.tensor(u[0])
        new_edges[1][i] = torch.tensor(u[1])   
        prev_id = idx[(u[0],u[1])]
        
        try:
            new_weights[prev_id] = graph.edge_weight[prev_id]
        except Exception:
            pass
        try:
            new_atrs[prev_id] = graph.edge_attr[prev_id]
            new_weights[prev_id] = graph.edge_attr[prev_id]
        except Exception:
            pass

    return new_edges, new_weights, new_atrs

def ToLineGraph(graph, verbose = False):
    new_edges, new_weights, new_atrs = RemoveDoubleEdges(graph)
    graph.num_edges = len(new_edges[0])
    graph.edge_index = new_edges
    graph.edge_attr = new_atrs
    graph.edge_weight = torch.reshape(graph.edge_attr, (len(graph.edge_attr), 1))

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
    graph.node_features = graph.edge_weight[:]
    
    new_edgeWeights = torch.ones([len(graph.edge_index[0]),1])
    graph.edge_weight = new_edgeWeights.flatten()
    graph.edge_attr = new_edgeWeights.flatten()
    
    
    graph.pos = None
    graph.num_nodes = len(graph.node_features) 
    graph.num_edges = len(graph.edge_attr)
    adj = GenerateAdjList(graph)
    AugmentNodeFeatures(graph, adj)
    
    assert(len(graph.edge_index[0])==len(graph.edge_weight))
    assert(graph.num_nodes==len(graph.x)) 
    assert(graph.num_nodes==num_edges)
    
    return graph
                
        
































