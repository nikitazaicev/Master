import os.path as osp
import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import SuiteSparseMatrixCollection
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from blossom import maxWeightMatching
from LineGraphConverter import ToLineGraph
from DataLoader import LoadData
torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.version.cuda)


dataset = SuiteSparseMatrixCollection('/Data', 'Newman', 'netscience',transform=NormalizeFeatures())

train_data = dataset[0]
train_data.edge_weight=train_data.edge_attr.unsqueeze(1)
train_data.node_features=torch.ones(train_data.num_nodes,1)

print("Before Line Graph Transform:", train_data)
print(f'Number of nodes: {train_data.num_nodes}')
print(f'Number of edges: {train_data.num_edges}')
print(f'edge_weight: {train_data.edge_weight}')
print(f'node_features: {train_data.node_features}')
print(f'edges: {train_data.edge_index}')

print("Blossom matching")
blossominput = []
for i in range(len(train_data.edge_index[0])):
    blossominput.append(
        (train_data.edge_index[0][i].item(),
         train_data.edge_index[1][i].item(),
         train_data.edge_weight[i][0].item()))

y = maxWeightMatching(blossominput)

print(f'matching: {y[:5]}')
print(f'count: {len(y)}')
print(f'from node: {train_data.edge_index[0][:5]}')
print(f'to node: {train_data.edge_index[1][:5]}')

original_graph = train_data.clone().to(device)
line_graph = ToLineGraph(train_data, verbose = False)

print("Line Graph Convert:", line_graph)

classes = []
total_matches = 0
for i in range(train_data.num_nodes):
    from_node = original_graph.edge_index[0][i].item()
    to_node = original_graph.edge_index[1][i].item()
    if y[from_node] == to_node:
        classes.append(1)
        total_matches += 1
    else:
        classes.append(0)
    
train_data.y = torch.IntTensor(classes)
print(f'classes: {train_data.y}')
print(f'Total matches: {total_matches}')
print(f'classes length: {len(train_data.y)}')

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.node_features, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
model = GCN().to(device)
train_data = train_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.010)#, weight_decay=0.01)

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(train_data)

    y = train_data.y.type(torch.LongTensor)
    y = y.to(device)
    loss = F.nll_loss(out, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'epoch{epoch+1}/100, loss={loss.item():.4f}')
  

  
model.eval()
pred = model(train_data)
correct = 0

top = torch.max(pred,1)
print("TOTAL EDGE CANDIDATES", torch.count_nonzero(top.indices))
print("TOP", top)
print("TRUTH", y)  

sorted_pred = torch.sort(top.values)                       
print("SORTED", sorted_pred)  

print("GREEDY PICK")
picked_edges = {(-1,-1)}
for i, sorted_i in enumerate(sorted_pred.indices):
    #print("edge number = ", sorted_i)
    from_node = original_graph.edge_index[0][sorted_i].item()
    to_node = original_graph.edge_index[1][sorted_i].item()
    if top.indices[sorted_i] == 1 and (from_node, to_node) not in picked_edges:
        #print("PICKED edge number = ", sorted_i)
        #print("PICKED edge = ", (from_node, to_node))
        picked_edges.add((from_node, to_node))

print("ALL EDGES: ", original_graph.edge_index)
print("GREEDY PICKED EDGES: ", picked_edges, "LEN =", len(picked_edges))
true_edges_idx = (y == 1).nonzero(as_tuple=True)[0]
true_edges = [(-1,-1)]
for idx in true_edges_idx:
    from_node = original_graph.edge_index[0][idx].item()
    to_node = original_graph.edge_index[1][idx].item()
    true_edges.append((to_node, from_node))        
print("TRUE EDGES: ", true_edges_idx, "LEN =", len(true_edges))

for i in range(len(y)):
    from_node = original_graph.edge_index[0][i].item()
    to_node = original_graph.edge_index[1][i].item()
    picked_contains = ((from_node, to_node) in picked_edges or (to_node, from_node) in picked_edges)
    conditionMet = (y[i] == 1 and picked_contains 
                    or y[i] == 0 and not picked_contains)

    if conditionMet:
        correct+=1


print(f'correct: {correct}')  
print(f'y: {len(train_data.y)}')  

acc = int(correct) / int(len(train_data.y))
print(f'Total Accuracy: {acc:.4f}')

acc = int(len(picked_edges)) / int(len(true_edges))
print(f'True Positives: {acc:.4f}')    

acc = int(len(y) - len(true_edges)) / int(len(y) - len(picked_edges))
print(f'True Negatives: {acc:.4f}')    
