import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.utils import negative_sampling

class MyGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(5, 320, aggr="max")
        self.conv2 = GCNConv(320, 160, aggr="max")
        # self.conv3 = GCNConv(640, 640, aggr="max")
        self.lin = Linear(160, 2)

    def forward(self, data):        
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        identity = F.relu(x)
        
        x = self.conv2(identity, edge_index)
        #x = F.relu(x)
        
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)

        # x = torch.cat((x,identity),1)
        x = self.lin(x)
        
        # return F.log_softmax(x, dim=1) #x
        return x
    
class EdgeClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(160, 320)
        self.lin2 = Linear(320, 2)
        
    def forward(self, nodeEmbed):        
        
        x = self.lin1(nodeEmbed)
        x = F.relu(x)
        
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)
    
    
    
class MyGCNEdge(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(5, 320)
        self.conv2 = GCNConv(320, 160)
        # self.conv3 = GCNConv(640, 640, aggr="max")
        self.embed = Linear(160, 80)
        self.lin1 = Linear(160, 320)
        self.lin2 = Linear(320, 2)

    def forward(self, data):        
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        
        x = self.conv1(x, edge_index, edge_weight)
        identity = F.relu(x)
        
        x = self.conv2(identity, edge_index, edge_weight)
        #x = F.relu(x)
        
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)

        # x = torch.cat((x,identity),1)
        x = self.embed(x)
        return x
    
    
    
    
    
    
    
    
    
    
    
    
    