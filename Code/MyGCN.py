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
        self.conv1 = GCNConv(5, 640, aggr="max")
        self.conv2 = GCNConv(640, 640, aggr="max")
        # self.conv3 = GCNConv(640, 640, aggr="max")
        self.lin = Linear(640, 2)

    def forward(self, data):        
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        identity = F.relu(x)
        
        x = self.conv2(identity, edge_index)
        x = F.relu(x)
        
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)

        x = torch.cat((x,identity),1)
        x = self.lin(x)
        
        # return F.log_softmax(x, dim=1) #x
        return x
    
class EdgeClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(161, 320)
        self.lin2 = Linear(320, 2)
    
    def embedEdges(self, nodeEmbed, graph):
        x_src, x_dst = nodeEmbed[graph.edge_index[0]], nodeEmbed[graph.edge_index[1]]
        
        edgeFeats = torch.zeros([len(graph.edge_index[0]),1]).to(device)
        uniqueEdges = set()
        for i in range(len(graph.edge_index[0])):
            from_node = graph.edge_index[0][i]
            to_node = graph.edge_index[1][i]
            
            if (from_node, to_node) in uniqueEdges: continue
            uniqueEdges.add((from_node, to_node))
            w = graph.edge_weight[i]            
            edgeFeats[i][0] = 1      
        
        edgeEmbed = torch.cat([x_src, edgeFeats, x_dst], dim=-1)        
        return edgeEmbed
    
    def forward(self, edgeEmbed):        
        
        x = self.lin1(edgeEmbed)
        x = F.relu(x)
        
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
    
class MyGCNEdge(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(7, 640)
        self.conv2 = GCNConv(640, 640)
        # self.conv3 = GCNConv(640, 640, aggr="max")
        self.embed = Linear(1280, 80)

    def forward(self, data):        
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = self.conv1(x, edge_index, edge_weight)
        identity = F.relu(x)
        x = self.conv2(identity, edge_index, edge_weight)
        x = F.relu(x)
        
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        x = torch.cat((x,identity),1)
        x = self.embed(x)
        return x

class MySAGEEdge(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(7, 640)
        self.conv2 = SAGEConv(640, 640)
        # self.conv3 = SAGEConv(640, 640, aggr="max")
        self.embed = Linear(640, 80)

    def forward(self, data):        
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        
        x = self.conv1(x, edge_index, edge_weight)
        identity = F.relu(x)
        
        x = self.conv2(identity, edge_index, edge_weight)
        x = F.relu(x)
        
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)

        #x = torch.cat((x,identity),1)
        x = self.embed(x)
        return x
    
def GNNMatching(gnn, classifier, graph, tresholdP = 0.5, tresholdN = 0.0, verbose = False):
    picked_edges, picked_nodes, droppedNodes = set(), set(), set()
    weightSum = 0
    step = 1

    while (graph.num_nodes-2*len(picked_edges)) > 2:
        if verbose: print("Step - ", step)
    
        
        out = gnn(graph)
        out = classifier(classifier.embedEdges(out,graph))  
        pred = torch.exp(out)
        
        scores = []
        for p in pred:
            if p[0] > p[1]: scores.append(0.0)
            else: scores.append(p[1])      
        scores = torch.FloatTensor(scores)
        

        #print(torch.max(scores))
        
        #weight, originalEdgeIds, pickedEdges = gp.GreedyScores(scores, val_graph, val_original_copy)
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
        step += 1
    return graph, weightSum, droppedNodes
    
    
    
    
    
    
    
    
    
    
    