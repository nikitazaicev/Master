import torch
import ReductionsManager as rm

torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def GreedyMatchingOrig(graph, prevEdgeIndeces=None, verbose=False):
    
    totalWeight = 0
    if prevEdgeIndeces is None: pickedEdgeIndeces = set()
    else: pickedEdgeIndeces = prevEdgeIndeces
    pickedPairs = set()
    takenNodes = set()
    edge_weights = graph.edge_weight.flatten()

    sorted_edges = torch.sort(edge_weights, descending=True)
    edge_index = graph.edge_index
    for i, original_index in enumerate(sorted_edges.indices):
        original_index = original_index.item()
        from_node = edge_index[0][original_index].item()
        to_node = edge_index[1][original_index].item()
        if (from_node not in takenNodes 
            and to_node not in takenNodes
            and original_index not in pickedEdgeIndeces):
            takenNodes.add(from_node)
            takenNodes.add(to_node)
            pickedEdgeIndeces.add(original_index)
            pickedPairs.add((from_node,to_node))
            totalWeight += edge_weights[original_index].item()
        
    for i in range(len(graph.edge_index[0])):
        from_node = graph.edge_index[0][i].item()
        to_node = graph.edge_index[1][i].item()
        if (from_node, to_node) in pickedPairs or (to_node, from_node) in pickedPairs: 
            pickedEdgeIndeces.add(i)    
            
    if verbose: print(f"Greedy matching total weight result = {totalWeight:.4f}")
    if verbose: print(f"In total {len(pickedEdgeIndeces)} out of {len(edge_weights)} edges were picked")
    if verbose: print(f"{len(takenNodes)} out of {graph.num_nodes} nodes in the matching")
    return totalWeight, pickedEdgeIndeces 

def GreedyMatchingTargets(graph):
    target = [0]*graph.num_edges
    totalWeight, pickedEdgeIndeces = GreedyMatchingOrig(graph)
    for idx in pickedEdgeIndeces:      
        target[idx] = 1    
    
    return torch.LongTensor(target).to(device)

def GreedyMatchingLine(graph):
    
    totalWeight = 0

    pickedEdgeIndeces = set()

    sorted_edges = torch.sort(graph.x[:, [0]].flatten(), descending=True)    
    
    adj = rm.GenerateAdjList(graph)
    count = 0
    for i, original_index in enumerate(sorted_edges.indices):
        original_index = original_index.item()
        
        if (original_index not in pickedEdgeIndeces):
            pickedEdgeIndeces.add(original_index)
            totalWeight += graph.x[original_index][0].item()
            pickedEdgeIndeces.update(adj[original_index])
            count += 1

    print(f"Greedy matching total weight result = {totalWeight:.4f}")
    print(f"In total {count} out of {len(sorted_edges.indices)} edges were picked")
    return totalWeight, pickedEdgeIndeces 

def RandomMatching(graph):
    
    totalWeight = 0.0
    pickedEdgeIndeces = []
    takenNodes = set()
    
    edge_weights = graph.edge_weight
    perm = torch.randperm(len(graph.edge_index[0]))
    edge_index = graph.edge_index
    
    for i in perm:
        from_node = edge_index[0][i].item()
        to_node = edge_index[1][i].item()
        if from_node not in takenNodes and to_node not in takenNodes:
            takenNodes.add(from_node)
            takenNodes.add(to_node)
            pickedEdgeIndeces.append(i)            
            totalWeight += edge_weights[i].item()
                 
    print(f"Random matching total weight result = {totalWeight:.4f}")
    print(f"In total {len(pickedEdgeIndeces)} out of {len(edge_weights)} edges were picked")
    print(f"{len(takenNodes)} out of {graph.num_nodes} nodes in the matching")
    return totalWeight, pickedEdgeIndeces 

def GreedyScores(pred, graph, original_g, threshold = 0.5):
    picked_nodes = set()
    picked_edgeIds = set()
    weightSum = 0
    sorted_pred = torch.sort(pred, descending=True)

    for i, sorted_i in enumerate(sorted_pred.indices):
        sorted_i = sorted_i
        from_node = original_g.edge_index[0][sorted_i].item()
        to_node = original_g.edge_index[1][sorted_i].item()
        
        if (sorted_pred.values[i] >= threshold 
            and (from_node not in picked_nodes 
            and to_node not in picked_nodes)):
            weightSum += original_g.edge_attr[sorted_i]
            picked_edgeIds.add(sorted_i.item())
            picked_nodes.add(from_node)
            picked_nodes.add(to_node)
    picked_neighbors = set()
    for i in range(len(graph.edge_index[0])):
        from_node = graph.edge_index[0][i].item()
        to_node = graph.edge_index[1][i].item()
        if from_node in picked_edgeIds:
            picked_neighbors.add(to_node)
        if to_node in picked_edgeIds:
            picked_neighbors.add(from_node)
    
    excluded_edgeIds = set()    
    excluded_edgeIds.update(picked_edgeIds)                
    excluded_edgeIds.update(picked_neighbors)
    return weightSum, excluded_edgeIds, len(picked_edgeIds)

def GreedyScoresOriginal(pred, original_g, threshold = 0.5, dropThreshold = 0.0):
    picked_nodes, dropped_nodes = set(), set()
    weightSum = 0
    sorted_pred = torch.sort(pred, descending=True)

    picked_edges = set()
    for i, sorted_i in enumerate(sorted_pred.indices):
        from_node = original_g.edge_index[0][sorted_i].item()
        to_node = original_g.edge_index[1][sorted_i].item()
        
        if (sorted_pred.values[i] >= threshold 
            and (from_node not in picked_nodes 
            and to_node not in picked_nodes)):
            weightSum += original_g.edge_weight[sorted_i][0]
            picked_nodes.add(from_node)
            picked_nodes.add(to_node)
            picked_edges.add(sorted_i)
        
        if (sorted_pred.values[i] < dropThreshold):
            picked_nodes.add(from_node)
            picked_nodes.add(to_node)
            picked_edges.add(sorted_i)
            dropped_nodes.add(from_node)
            dropped_nodes.add(to_node)

    
    return weightSum, picked_edges, picked_nodes, dropped_nodes

