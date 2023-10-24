import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import GreedyPicker as gp
import DataLoader as dl
import teststuff
from MyGCN import MyGCN


torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current CUDA version: ", torch.version.cuda, "\n")

original, converted_dataset, target = dl.LoadData(1000)

#original, converted_dataset, target = dl.LoadTestData()

print("Data stats")
print("----------------")
print(original[0])
print(f'Total graphs = {len(original)}')
#print(f'Undirected original = {original[0].is_undirected()}')
#print(f'Undirected converted = {converted_dataset[0].is_undirected()}')
print(f'Nodes in original graph = {original[0].num_nodes}')
print(f'Edges in original graph = {len(original[0].edge_index[0])}')
print(f'Nodes in line graph = {converted_dataset[0].num_nodes}')
print(f'Edges in line graph = {len(converted_dataset[0].edge_index[0])}')
assert((len(converted_dataset[0].edge_weight)==len(converted_dataset[0].edge_index[0])
       and len(converted_dataset[0].edge_index[0]==len(converted_dataset[0].edge_attr))))
print("-------------------\n")

train_test_split = int(0.8*len(original))

original_graphs = original[:train_test_split]#[:1]#[:train_test_split]
train_data = converted_dataset[:train_test_split]#[:1]#[:train_test_split1]
y = target[:train_test_split]#[:1]#[:train_test_split]

val_original_graphs = original[train_test_split:]#[:1]#[train_test_split:]
val_data = converted_dataset[train_test_split:]#[:1]#[train_test_split:]
val_y = target[train_test_split:]#[:1]#[train_test_split:]

#val_original_graphs, val_data, val_y = dl.LoadValExample()

print("Assigning target classes")
def AssignTargetClasses(data, original, targetY):
    classes = [[] for i in range(len(data))]         
    for i, graph in enumerate(data):
        for j in range(graph.num_nodes):
            from_node = original[i].edge_index[0][j].item()
            to_node = original[i].edge_index[1][j].item()
            if targetY[i][from_node] == to_node:
                classes[i].append(1)
            else:
                classes[i].append(0)
    
    for i, graph in enumerate(data):        
        graph.y = torch.LongTensor(classes[i]).to(device)
    
print("Saveing visualization file")
AssignTargetClasses(train_data, original_graphs, y)
AssignTargetClasses(val_data, val_original_graphs, val_y)
dl.VisualizeConverted(train_data[0])
dl.VisualizeOriginal(original_graphs[0])
print("------------------- \n")

print("STARING TRAINING")
print("-------------------")

torch.manual_seed(123)
model = MyGCN().to(device)

loader = DataLoader(train_data, batch_size=1, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#, weight_decay=0.01)
classWeights = torch.FloatTensor([0.1,0.9]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=classWeights)

epochs = 10
torch.manual_seed(123)
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
val_original_copy = val_original_graphs[0].to(device)
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


picked_edges = set()
picked_nodes = set()
weightSum = 0
step = 1
while (val_original.num_nodes-2*len(picked_edges)) > 2:
    print("Step - ", step)
   
    pred = torch.exp(model(val_graph.to(device)))
    scores = []
    for p in pred:
        if p[0] > p[1]: scores.append(0.0)
        else: scores.append(p[1])      
    scores = torch.FloatTensor(scores)
    
    weight, originalEdgeIds, pickedEdges = gp.GreedyScores(scores, val_graph, val_original_copy)
    weightSum += weight
    weight = 0
    print("Total original edges picked = ", pickedEdges)
    print("Total original edges excluded (with neighbors) = ", len(originalEdgeIds))
    print("Current weight sum = ", weightSum)
    if (pickedEdges == 0) : break

    print("Removing picked nodes...")
    print("Total nodes in converted graph = ", len(val_graph.x))
    val_original_copy, val_graph = teststuff.ReduceGraph(val_original_copy, val_graph, originalEdgeIds)
    print("Total nodes in converted graph remains = ", len(val_graph.x))
    if val_graph.num_nodes <= 0: break    
    step += 1


print("Finishing remaining matching with normal greedy.")
weightRes, pickedEdgeIndeces = gp.GreedyMatchingLine(val_graph)
weightSum += weightRes
print("Total original edges picked = ", len(pickedEdgeIndeces))
print("Current weight sum = ", weightSum)
print("No more matching posibilities left.")
print("------------------- \n") 

print("STARTING STANDARD GREEDY SEARCH.")
print("-------------------")
weightGreedy, pickedEdgeIndeces = gp.GreedyMatchingOrig(val_original)
print("------------------- \n") 

print("STARTING RANDOM MATCHING.")
print("-------------------")
weightRandom, pickedEdgeIndeces = gp.RandomMatching(val_original)
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
print("Optimal amount of matches = ", opt_matches)

correct = 0
correct_picked = 0
false_picked = 0
correct_dropped = 0 
false_dropped = 0    
for i in range(len(val_y_item)):
    from_node = val_original.edge_index[0][i].item()
    to_node = val_original.edge_index[1][i].item()
    
    picked_contains = (i in picked_edges)

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

# tp = correct_picked/opt_matches
# tf = correct_dropped/opt_drops
# fp = false_picked/opt_drops
# fn = false_dropped/opt_matches

# try:
#     precision = tp/(tp+fp)
#     recall = tp/(tp+fn)
#     f1score = (2*precision*recall)/(precision+recall)    
# except Exception:
#     f1score = 0.0

# print(f'TRUE POSITIVE (opt): {correct_picked}/{opt_matches}')
# print(f'TRUE NEGATIVE (opt): {correct_dropped}/{opt_drops}')
# print(f'FALSE POSITIVE (opt): {false_picked}/{opt_drops}')
# print(f'FALSE NEGATIVE (opt): {false_dropped}/{opt_matches}')
# print(f'F1 SCORE (opt): {f1score:.4f}')

# acc = int(correct) / int(len(val_y_item))
# print(f'Total ACCURACY: {acc:.2f}')
# print("------------------- \n") 

print(f'Total WEIGHT out of optimal: {weightSum:.2f}/{weightMax:.2f} ({100*(weightSum/weightMax):.1f}%) ')
print(f'Total WEIGHT out of standard greedy: {weightSum:.2f}/{weightGreedy:.2f} ({100*(weightSum/weightGreedy):.1f}%)')
print(f'Total WEIGHT out of random matching: {weightSum:.2f}/{weightRandom:.2f} ({100*(weightSum/weightRandom):.1f}%)')

