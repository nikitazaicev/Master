import torch
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import SuiteSparseMatrixCollection


def PrintInfo(_data, verbose = True):
    if not verbose: return
    print(_data)
    print(f'Number of nodes: {_data.num_nodes}')
    print(f'Number of edges: {_data.num_edges}')
    print(f'edges: {_data.edge_index}')

def ToLineGraph(graph, edge_weight, verbose = False):
    
    if verbose: print("-------- BEFORE --------")
    PrintInfo(graph)
    
    num_edges = 0
    num_nodes = 0
    newNodes = {}
    bucket = {new_list: [] for new_list in range(graph.num_edges)}
    newEdges = set()
    
    for i in range(len(edge_weight)):
        
        from_node = graph.edge_index[0][i]
        to_node = graph.edge_index[1][i]
            
        newNodes[num_nodes] = (torch.tensor(num_nodes),from_node,to_node)

        bucket[from_node.item()].append(num_nodes)
        bucket[to_node.item()].append(num_nodes)

        num_nodes = num_nodes+1
    
    for i in range(len(edge_weight)):

        to_node = graph.edge_index[1][i]
        if to_node.item() in bucket:
            for n in bucket[to_node.item()]:
                for m in bucket[to_node.item()]:
                    if n!=m: 
                        newEdges.add((n,m)) # and (m,n) not in newEdges
                        
    newEdges = sorted(newEdges, key=lambda x : x[0])
    
    num_edges = len(newEdges)
    newEdgesTensor = torch.zeros([2,num_edges], dtype=torch.int)
    
    for idx, e in enumerate(newEdges):
        newEdgesTensor[0][idx] = e[0]
        newEdgesTensor[1][idx] = e[1]
        
    new_edgeWeights = torch.ones([num_edges,1])
    graph.edge_index = newEdgesTensor
    graph.node_features = edge_weight[:]
    
    graph.edge_weight = new_edgeWeights.flatten()
    graph.edge_attr = new_edgeWeights.flatten()
    
    graph.x = graph.node_features.resize(len(graph.node_features),1)
    graph.pos = None
    graph.num_nodes = len(graph.node_features) 
    graph.num_edges = len(graph.edge_attr) 
    
    if verbose:
        print("-------- AFTER --------")
        PrintInfo(graph)

    return graph

