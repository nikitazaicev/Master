import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear, SAGEConv
from torch_geometric.utils import negative_sampling
import GreedyPicker as gp
import ReductionsManager as rm
from torch.nn.functional import normalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 64)#, aggr="sum")
        self.conv2 = GCNConv(64, 64)#, aggr="sum")
        self.conv3 = GCNConv(64, 64)#, aggr="sum")
        self.lin = Linear(64, 2)

    def forward(self, data):        
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        identity = F.relu(x)
        x = identity
        
        #x = self.conv2(identity, edge_index)
        #x = F.relu(x)
        
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)

        #x = torch.cat((x,identity),1)
        x = self.lin(x)
        
        return F.log_softmax(x, dim=1) #x
        
class EdgeClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(161, 32)
        #self.lin2 = Linear(32, 32)
        #self.lin4 = Linear(32, 32)
        #self.lin5 = Linear(32, 32)
        self.lin3 = Linear(32, 2)
    
    def embedEdges(self, nodeEmbed, graph):
        x_src, x_dst = nodeEmbed[graph.edge_index[0]], nodeEmbed[graph.edge_index[1]]
        
        edgeFeats = torch.ones([len(graph.edge_index[0]),1]).to(device)
        #uniqueEdges = set()
        # for i in range(len(graph.edge_index[0])):
        #     from_node = graph.edge_index[0][i]
        #     to_node = graph.edge_index[1][i]
            
        #     if (from_node, to_node) in uniqueEdges: continue
        #     uniqueEdges.add((from_node, to_node))
        #     w = graph.edge_weight[i]            
        #     edgeFeats[i][0] = 1      
        
        edgeEmbed = torch.cat([x_src, edgeFeats, x_dst], dim=-1)        
        return edgeEmbed
    
    def forward(self, edgeEmbed):        
        
        x = self.lin1(edgeEmbed)
        x = F.relu(x)
        
        x = self.lin2(x)
        x = F.relu(x)

        #x = self.lin4(x)
        #x = F.relu(x)
        
        #x = self.lin5(x)
        #x = F.relu(x)
        
        x = self.lin3(x)
        return F.log_softmax(x, dim=1)
    
class MyGCNEdge(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(5, 240, aggr="max")
        self.conv2 = GCNConv(240, 240, aggr="max")
        #self.conv3 = GCNConv(64, 64)
        #self.conv4 = GCNConv(64, 64)
        #self.embed = Linear(1280, 80)
        self.embed = Linear(240, 80)

    def forward(self, data):        
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = self.conv1(x, edge_index, edge_weight)
        identity = F.relu(x)
        
        x = self.conv2(identity, edge_index, edge_weight)
        x = F.relu(x)

        #x = self.conv4(x, edge_index, edge_weight)
        #x = F.relu(x)
        
        #x = self.conv3(x, edge_index, edge_weight)
        #x = F.relu(x)
        
        #x = torch.cat((x,identity),1)
        x = self.embed(x)
        return x
    
def GNNMatching(gnn, classifier, graph, tresholdP = 0.5, tresholdN = 0.0, verbose = False, test = set()):
    picked_edges, picked_nodes, droppedNodes = set(), set(), set()
    weightSum = 0
    step = 1

    while (graph.num_nodes-2*len(picked_edges)) > 2 and step <= 2:
        if verbose: print("Step - ", step)
        step += 1
        
        out = gnn(graph)
        out = classifier(classifier.embedEdges(out,graph))  
        pred = torch.exp(out)
        
        scores = []
        for p in pred:
            if p[0] > p[1]: scores.append(0.0)
            else: scores.append(p[1])      
        scores = torch.FloatTensor(scores)
        #print(scores)
        weight, originalEdgeIds, picked_nodes, dropNodes = gp.GreedyScoresOriginal(scores, graph, tresholdP, tresholdN)
        
        droppedNodes.update(dropNodes)
        pickedEdges = len(originalEdgeIds)
        
        weightSum += weight
        weight = 0
        # print("Total original edges removed = ", pickedEdges)
        if verbose: print("Current weight sum = ", weightSum)
        if (pickedEdges == 0) : break
    
        #val_original_copy, val_graph = teststuff.ReduceGraph(val_original_copy, val_graph, originalEdgeIds)
        graph = rm.ReduceGraphOriginal(graph, picked_nodes)
        if verbose: print("Total nodes in converted graph remains = ", len(graph.x))
        if graph.num_nodes <= 0: break    
        
    return graph, weightSum, picked_nodes, step
    
def GNNMatchingLine(gnn, line, graph, tresholdP = 0.5):
    picked_edges, picked_nodes, droppedNodes = set(), set(), set()
    weightSum = 0
    step = 1

    while (line.num_nodes-2*len(picked_edges)) > 2 and step <= 2:
        step += 1
        
        out = gnn(line)  
        pred = torch.exp(out)
        
        scores = []
        for p in pred:
            if p[0] > p[1]: scores.append(0.0)
            else: scores.append(p[1])      
        scores = torch.FloatTensor(scores)
        #print(scores)
        weightSum, picked_edges, pickedEdges = gp.GreedyScores(scores, line, graph, tresholdP)
        
        # print("Total original edges removed = ", pickedEdges)
        if (pickedEdges == 0) : break
    
        #val_original_copy, val_graph = teststuff.ReduceGraph(val_original_copy, val_graph, originalEdgeIds)
        line = rm.ReduceGraph(line, picked_edges)
        if line.num_nodes <= 0: break    
        
    return line, weightSum, picked_nodes, step    
    
    
    
    
    
    
    
    
    