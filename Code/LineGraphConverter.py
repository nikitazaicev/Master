import torch
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import SuiteSparseMatrixCollection


def PrintInfo(_data, verbose = True):
    if not verbose: return
    print(_data)
    print(f'Number of nodes: {_data.num_nodes}')
    print(f'Number of edges: {_data.num_edges}')
    #print(f'edge_weight: {_data.edge_weight}')
    print(f'node_features: {_data.node_features}')
    print(f'edges: {_data.edge_index}')

def ToLineGraph(graph, verbose = False):
    
    if verbose: print("-------- BEFORE --------")
    PrintInfo(graph)
    
    num_edges = 0
    num_nodes = 0
    newNodes = {}
    bucket = {new_list: [] for new_list in range(graph.num_edges)}
    newEdges = set()
    
    for i in range(len(graph.edge_weight)):
        
        from_node = graph.edge_index[0][i]
        to_node = graph.edge_index[1][i]

        # if verbose: 
        #     print("Current node = " + str(num_nodes))        
        #     print("From node = " + str(from_node))
        #     print("To node = " + str(to_node))
        #     print("Edges = " + str(num_edges))
            
        newNodes[num_nodes] = (torch.tensor(num_nodes),from_node,to_node)

        bucket[from_node.item()].append(num_nodes)
        bucket[to_node.item()].append(num_nodes)

        num_nodes = num_nodes+1
    
    for i in range(len(graph.edge_weight)):

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
    
    #print("New edges", newEdgesTensor)
    
    new_edgeWeights = torch.ones([num_edges,1])
    graph.edge_index = newEdgesTensor
    graph.node_features = graph.edge_weight[:]
    
    graph.edge_weight = new_edgeWeights
    graph.edge_attr = new_edgeWeights.flatten()
    graph.num_nodes = len(graph.node_features) 
    graph.num_edges = len(graph.edge_weight) 
    
    if verbose:
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




