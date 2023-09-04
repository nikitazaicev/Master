import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from DataLoader import LoadData
from MyGCN import MyGCN

torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.version.cuda)


# dataset = SuiteSparseMatrixCollection('/Data', 'Newman', 'netscience', transform=NormalizeFeatures())
# print("dataset", len(dataset))

# train_data = dataset[0]
# train_data.edge_weight=train_data.edge_attr.unsqueeze(1)
# train_data.node_features=torch.ones(train_data.num_nodes,1)

# print("Blossom matching")
# blossominput = []
# for i in range(len(train_data.edge_index[0])):
#     blossominput.append(
#         (train_data.edge_index[0][i].item(),
#           train_data.edge_index[1][i].item(),
#           train_data.edge_weight[i][0].item()))

# y = maxWeightMatching(blossominput)
# print(y)
# original_graph = train_data.clone().to(device)
# line_graph = ToLineGraph(train_data, verbose = False)

original, converted_dataset, target = LoadData(5)
train_test_split = int(0.8*len(original))
original_graphs = original[:train_test_split]
train_data = converted_dataset[:train_test_split]
val_data = converted_dataset[train_test_split:]
y = target[:train_test_split]

print("Assigning target classes")
classes = [[] for i in range(len(train_data))]
for i, graph in enumerate(train_data):
    for j in range(graph.num_nodes):
        from_node = original_graphs[i].edge_index[0][j].item()
        to_node = original_graphs[i].edge_index[1][j].item()
        if y[i][from_node] == to_node:
            classes[i].append(1)
        else:
            classes[i].append(0)

for i, graph in enumerate(train_data):        
    graph.y = torch.IntTensor(classes[i])

    
model = MyGCN().to(device)
loader = DataLoader(train_data, batch_size=32, shuffle=True)

print("Staring training")

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
model.train()
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    
    for graphs in loader:
        graphs = graphs.to(device)
        out = model(graphs)
        y = graphs.y.type(torch.LongTensor)
        y = y.to(device)
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    if (epoch + 1) % 10 == 0:
        print(f'epoch{epoch+1}/100, loss={loss.item():.4f}')
  
print("Finished training")    

print("Starting evaluation")  
model.eval()
pred = model(val_data[0].to(device))
correct = 0

top = torch.max(pred,1)
print("TOTAL EDGE CANDIDATES", torch.count_nonzero(top.indices))
print("TOP", top)
print("TRUTH", y)  

sorted_pred = torch.sort(top.values)                       
print("SORTED", sorted_pred)  

print("GREEDY PICK")
picked_edges = {(-1,-1)}
picked_nodes = {-1}
weightSum = 0
for i, sorted_i in enumerate(sorted_pred.indices):
    from_node = original_graph.edge_index[0][sorted_i].item()
    to_node = original_graph.edge_index[1][sorted_i].item()
    if top.indices[sorted_i] == 1 and from_node not in picked_nodes and to_node not in picked_nodes: #and (from_node, to_node) not in picked_edges:
        weightSum += 2*original_graph.edge_weight[sorted_i]
        picked_edges.add((from_node, to_node))
        picked_edges.add((to_node, from_node))
        picked_nodes.add(from_node)
        picked_nodes.add(to_node)
        
for i, sorted_i in enumerate(sorted_pred.indices):
    from_node = original_graph.edge_index[0][sorted_i].item()
    to_node = original_graph.edge_index[1][sorted_i].item()
    if top.indices[sorted_i] == 0 and from_node not in picked_nodes and to_node not in picked_nodes:
        weightSum += 2*original_graph.edge_weight[sorted_i]
        picked_edges.add((from_node, to_node))
        picked_edges.add((to_node, from_node))
        picked_nodes.add(from_node)
        picked_nodes.add(to_node)
            

print("ALL EDGES: ", original_graph.edge_index)
print("GREEDY PICKED EDGES: ", picked_edges, "LEN =", len(picked_edges))

true_edges_idx = (y == 1).nonzero(as_tuple=True)[0]
true_edges = [(-1,-1)]

weightMax = 0
for idx in true_edges_idx:
    from_node = original_graph.edge_index[0][idx].item()
    to_node = original_graph.edge_index[1][idx].item()
    weightMax += original_graph.edge_weight[idx]
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

w = weightSum / weightMax
print(f'Total Weight: {w.item():.4f}')

acc = int(correct) / int(len(train_data.y))
print(f'Total Accuracy: {acc:.4f}')

acc = int(len(picked_edges)) / int(len(true_edges))
print(f'True Positives: {acc:.4f}')    

acc = int(len(y) - len(true_edges)) / int(len(y) - len(picked_edges))
print(f'True Negatives: {acc:.4f}')    
