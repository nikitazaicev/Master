import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear

class MyGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(5, 640, aggr="max")
        self.conv2 = GCNConv(640, 640, aggr="max")
        #self.conv3 = GCNConv(640, 640, aggr="max")
        self.lin = Linear(1280, 2)

    def forward(self, data):        
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        identity = F.relu(x)
        
        x = self.conv2(identity, edge_index)
        x = F.relu(x)
        
        #x = self.conv3(x, edge_index)
        #x = F.relu(x)

        x = torch.cat((x,identity),1)
        x = self.lin(x)
        
        
        
        return F.log_softmax(x, dim=1) #x