import os.path as osp
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import SuiteSparseMatrixCollection, KarateClub, Planetoid
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, train_test_split_edges, negative_sampling
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, TransformerConv, SAGEConv
import time
import dgl.function as fn
import dgl
import dgl.nn as dglnn
import numpy as np
from blossom import maxWeightMatching
torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.version.cuda)


dataset = SuiteSparseMatrixCollection('/Data', 'Newman', 'netscience',transform=NormalizeFeatures())

train_data = dataset[0]
train_data.edge_weight=train_data.edge_attr.unsqueeze(1)
train_data.node_features=torch.ones(train_data.num_nodes,1)

print(train_data)
print(f'Number of nodes: {train_data.num_nodes}')
print(f'Number of edges: {train_data.num_edges}')
print(f'edge_weight: {train_data.edge_weight}')
print(f'node_features: {train_data.node_features}')

blossominput = []
for i in range(len(train_data.edge_index[0])):
    blossominput.append(
        (train_data.edge_index[0][i].item(),
         train_data.edge_index[1][i].item(),
         train_data.edge_weight[i][0].item()))

y = maxWeightMatching(blossominput)

print(f'matching: {y[:5]}')
print(f'from node: {train_data.edge_index[0][:5]}')
print(f'to node: {train_data.edge_index[1][:5]}')

edge_classes = []
for i in range(len(train_data.edge_index[0])):
    if train_data.edge_index[1][i] == y[train_data.edge_index[0][i].item()]:
        edge_classes.append(1)
    else:
        edge_classes.append(0)
    
train_data.y = torch.IntTensor(edge_classes)
print(f'edge_classes: {train_data.y}')
print(f'edge_classes length: {len(train_data.y)}')

class EdgeGCN(torch.nn.Module):
    def __init__(self):
        super(EdgeGCN, self).__init__()
        aggr_type = 'sum'
        self.conv1 = GCNConv(1, 128, aggr=aggr_type)
        self.conv2 = GCNConv(128, 128, aggr=aggr_type )
        self.conv3 = GCNConv(128, 2, aggr=aggr_type )

    def forward(self, data):
        edge_index, edge_weight = data.edge_index, data.edge_weight
              
        x = self.conv1(edge_weight, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        
        return F.log_softmax(x, dim=1)
    
model = EdgeGCN().to(device)
train_data = train_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.010)#, weight_decay=0.01)

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(train_data)

    y = train_data.y.type(torch.LongTensor)
    y = y.to(device)
    print(f'predictions: {out[:10]}')
    print(f'predictions argmax 0: {torch.argmax(out[0])}')
    print(f'predictions argmax 1: {torch.argmax(out[1])}')
    print(f'ys: {y[:10]}')  
    loss = F.nll_loss(out, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'epoch{epoch+1}/100, loss={loss.item():.4f}')
    
model.eval()
pred = model(train_data)
for i in range(5000):
    if(torch.argmax(pred[i])==1):
        print(f'prediction: {torch.argmax(pred[i])}')  
        print(f'y: {train_data.y[i]}')  
correct = 0

for i in range(len(train_data.y)):
    if torch.argmax(pred[i]) == train_data.y[i]:
        correct+=1


print(f'correct: {correct}')  
print(f'y: {len(train_data.y)}')  

acc = int(correct) / int(len(train_data.y))
print(f'Accuracy: {acc:.4f}')
    
