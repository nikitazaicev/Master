from DataLoader import LoadData
import torch


def GreedyMatching(graph, edge_weights):
    
    totalWeight = 0
    pickedEdgeIndeces = []
    takenNodes = set()
    
    sorted_edges = torch.sort(edge_weights, descending=True)
    edge_index = graph.edge_index
    for i, original_index in enumerate(sorted_edges.indices):
        original_index = original_index.item()
        from_node = edge_index[0][original_index].item()
        to_node = edge_index[1][original_index].item()
        if from_node not in takenNodes and to_node not in takenNodes:
            takenNodes.add(from_node)
            takenNodes.add(to_node)
            pickedEdgeIndeces.append(original_index)
            totalWeight += edge_weights[original_index]
            
    print(f"Greedy matching total weight result = {totalWeight:.4f}")
    print(f"In total {len(pickedEdgeIndeces)} out of {len(edge_weights)} edges were picked")
    print(f"{len(takenNodes)} out of {graph.num_nodes} nodes in the matching")
    return totalWeight, pickedEdgeIndeces 


def GreedyScores(pred, graph, original_g, threshold = 0.5):
    picked_edges = set()
    picked_nodes = set()
    weightSum = 0
    sorted_pred = torch.sort(pred, descending=True)
    for i, sorted_i in enumerate(sorted_pred.indices):
        from_node = original_g.edge_index[0][sorted_i].item()
        to_node = original_g.edge_index[1][sorted_i].item()
        
        if (sorted_pred.values[i] >= threshold 
            and (from_node not in picked_nodes 
            or to_node not in picked_nodes)):
            
            weightSum += original_g.edge_attr[sorted_i]
            picked_edges.add((from_node, to_node))
            picked_nodes.add(from_node)
            picked_nodes.add(to_node)
            
    return weightSum, picked_nodes, picked_edges
