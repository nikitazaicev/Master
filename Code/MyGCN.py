import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear

class MyGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(2, 640, aggr="max")
        self.conv2 = GCNConv(640, 640, aggr="max")
        self.lin = Linear(640, 2)

    def forward(self, data):        
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.normalize(x, dim = 1)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = F.normalize(x, dim = 1)
        
        x = self.lin(x)
        #x = F.normalize(x, dim = 1)
        return F.log_softmax(x, dim=1) #x