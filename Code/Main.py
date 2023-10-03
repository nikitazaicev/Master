import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import GreedyPicker as gp
import DataLoader as dl
from MyGCN import MyGCN

torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current CUDA version: ", torch.version.cuda, "\n")

original, converted_dataset, target = dl.LoadData(1)

original, converted_dataset, target = dl.LoadTestData()

print("Data stats")
print("----------------")
print(original[0])
print(f'Undirected = {original[0].is_undirected()}')
print(f'Nodes in original graph = {original[0].num_nodes}')
print(f'Edges in original graph = {len(original[0].edge_index[0])}')
print(f'Nodes in line graph = {converted_dataset[0].num_nodes}')
print(f'Edges in line graph = {len(converted_dataset[0].edge_index[0])}')
print("-------------------\n")


train_test_split = int(0.8*len(original))

original_graphs = original[:1]#[:train_test_split]
train_data = converted_dataset[:1]#[:train_test_split]
y = target[:1]#[:train_test_split]

# val_original_graphs, val_data, val_y = dl.LoadValExample()

val_original_graphs = original[:1]#[train_test_split:]
val_data = converted_dataset[:1]#[train_test_split:]
val_y = target[:1]#[train_test_split:]
 
print("Assigning target classes")
def AssignTargetClasses(data, original):
    classes = [[] for i in range(len(data))]         
    for i, graph in enumerate(data):
        for j in range(graph.num_nodes):
            from_node = original[i].edge_index[0][j].item()
            to_node = original[i].edge_index[1][j].item()
            
            if y[i][from_node] == to_node:
                classes[i].append(1)
            else:
                classes[i].append(0)
    
    for i, graph in enumerate(train_data):        
        graph.y = torch.LongTensor(classes[i]).to(device)
    

AssignTargetClasses(train_data, original_graphs)
print("Saveing visualization file")
dl.VisualizeConverted(train_data[0])
dl.VisualizeOriginal(original[0])
print("------------------- \n")

print("STARING TRAINING")
print("-------------------")
model = MyGCN().to(device)

loader = DataLoader(train_data, batch_size=1, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.08)#, weight_decay=0.01)
classWeights = torch.FloatTensor([0.2,0.8]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=classWeights)


epochs = 100
model.train()
for epoch in range(epochs):
    
    for graphs in loader:
        
        optimizer.zero_grad()
        graphs = graphs.to(device)
        out = model(graphs).to(device)
        y = graphs.y.to(device)
        loss = F.nll_loss(out, y, weight=classWeights) #criterion(out, y) # 
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 10 == 0:
        print(f'epoch{epoch+1}/{epochs}, loss={loss.item():.4f}')

print("Successfully finished training")    
print("------------------- \n") 

print("STARING EVALUATION")  
print("-------------------")
model.eval()
val_original = val_original_graphs[0].to(device)
val_graph = val_data[0].to(device)
val_y_item = torch.LongTensor(val_y[0]).to(device)

classes = []
for j in range(val_graph.num_nodes):
    from_node = val_original.edge_index[0][j].item()
    to_node = val_original.edge_index[1][j].item()
    if val_y_item[from_node] == to_node:
        classes.append(1)
    else:
        classes.append(0)
val_y_item = torch.FloatTensor(classes).to(device)

print("GNN-score based greedy pick")

pred = torch.exp(model(val_graph))
scores = []
for p in pred:
    if p[0] > p[1]:
        scores.append(0.0)
    else:
        scores.append(p[1])      
scores = torch.FloatTensor(scores)
print("Prediction: ", pred[:10])
print("Scores: ", scores[:10])
print("Truth: ", val_y_item[:10])
picked_edges = set()
picked_nodes = set()
weightSum = 0
step = 1
while (val_original.num_nodes-len(picked_nodes)) > 2:
    print("Step - ", step)
    sort_class = 1
    weight, nodes, edges = gp.GreedyScores(scores, val_graph, val_original)
    if (len(edges)==0) : break
    picked_edges.update(edges)
    picked_nodes.update(nodes)
    weightSum += weight
    print("Weight sum = ", weightSum)
    print("Total picked nodes = ", len(picked_nodes))
    print("Total picked edges = ", len(picked_edges))
    print("Remaining nodes = ", val_original.num_nodes-len(picked_nodes))
    print("Remaining edges = ", val_original.num_nodes-len(picked_edges))
    print("Creating remaining graph...")
    break    
    step += 1

print("Remaining GNN scores all negative.")
print("------------------- \n") 

print("STARTING STANDARD GREEDY SEARCH.")
print("-------------------")
weightGreedy, pickedEdgeIndeces = gp.GreedyMatching(val_original, val_original.edge_attr)
print("------------------- \n") 

print("FINAL STATISTICS")
print("-------------------")
true_edges_idx = (val_y_item == 1).nonzero(as_tuple=True)[0]
true_edges = []

weightMax = 0
for idx in true_edges_idx:
    from_node = val_original.edge_index[0][idx].item()
    to_node = val_original.edge_index[1][idx].item()
    weightMax += val_original.edge_attr[idx]
    true_edges.append((to_node, from_node))     
opt_matches = len(true_edges_idx)
opt_drops = len(val_original.edge_attr) - len(true_edges_idx)
print("Optimal amount of matches = ", opt_matches )

correct = 0
correct_picked = 0
false_picked = 0
correct_dropped = 0 
false_dropped = 0    
print("picked_edges", picked_edges)
for i in range(len(val_y_item)):
    from_node = val_original.edge_index[0][i].item()
    to_node = val_original.edge_index[1][i].item()
    
    picked_contains = ((from_node, to_node) in picked_edges 
                       or (to_node, from_node) in picked_edges)

    if val_y_item[i] == 1 and picked_contains:
        correct_picked+=1
        correct+=1
    if val_y_item[i] != 1 and picked_contains:
        false_picked += 1
    if val_y_item[i] != 0 and not picked_contains:
        false_dropped += 1
    if val_y_item[i] == 0 and not picked_contains:
        correct_dropped+=1
        correct+=1

tp = correct_picked/opt_matches
tf = correct_dropped/opt_drops
fp = false_picked/opt_drops
fn = false_dropped/opt_matches

try:
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1score = (2*precision*recall)/(precision+recall)    
except Exception:
    f1score = 0.0

print(f'TRUE POSITIVE (opt): {correct_picked}/{opt_matches}')
print(f'TRUE NEGATIVE (opt): {correct_dropped}/{opt_drops}')
print(f'FALSE POSITIVE (opt): {false_picked}/{opt_drops}')
print(f'FALSE NEGATIVE (opt): {false_dropped}/{opt_matches}')
print(f'F1 SCORE (opt): {f1score}')


print(f'Total WEIGHT out of optimal: {weightSum:.2f}/{weightMax:.2f} ')
print(f'Total WEIGHT out of standard greedy: {weightSum:.2f}/{weightGreedy:.2f} ')

acc = int(correct) / int(len(val_y_item))
print(f'Total ACCURACY: {acc:.2f}')
print("------------------- \n") 