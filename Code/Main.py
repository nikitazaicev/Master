import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import GreedyPicker as gp
import DataLoader as dl
import ReductionsManager as rm
import MyGCN
import pickle
import random

torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current CUDA version: ", torch.version.cuda, "\n")

original, converted_dataset, target = dl.LoadData(1000)

print("Data stats")
print("----------------")
print(original[0])
print(f'Total graphs = {len(original)}')
print(f'Nodes in original graph = {original[0].num_nodes}')
print(f'Edges in original graph = {len(original[0].edge_index[0])}')
print(f'Nodes in line graph = {converted_dataset[0].num_nodes}')
print(f'Edges in line graph = {len(converted_dataset[0].edge_index[0])}')
assert((len(converted_dataset[0].edge_weight)==len(converted_dataset[0].edge_index[0])
       and len(converted_dataset[0].edge_index[0]==len(converted_dataset[0].edge_attr))))
print("-------------------\n")

train_test_split = int(0.8*len(original))
random.seed(123)

original_graphs = original[:train_test_split]
train_data = converted_dataset[:train_test_split]
y = target[:train_test_split]

val_original_graphs = original[train_test_split:]
val_data = converted_dataset[train_test_split:]
val_y = target[train_test_split:]

#val_original_graphs, val_data, val_y = dl.LoadValExample()
   
#print("Saveing visualization file")
#dl.VisualizeConverted(val_data[0])
#dl.VisualizeOriginal(val_original_graphs[0])
#print("------------------- \n")

print("STARING TRAINING")
print("-------------------")

torch.manual_seed(123)

model = MyGCN.MyGCNEdge().to(device)
#model = MyGCN.MyGCN().to(device)

classifier = MyGCN.EdgeClassifier().to(device)

#loader = DataLoader(train_data, batch_size=1, shuffle=True)
loader = DataLoader(original_graphs, batch_size=1, shuffle=True)
optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.001)#, weight_decay=0.0001)
classWeights = torch.FloatTensor([0.1,0.9]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=classWeights)

modelLoaded = False
try:
    with open('data/MyModel.pkl', 'rb') as file:
        print("Saved model found, skipping training")
        model = pickle.load(file)
    with open('data/MyModelClass.pkl', 'rb') as file:
        classifier = pickle.load(file)      
        modelLoaded = True
except Exception: print("No model found, training new model")

if not modelLoaded:
    EPOCHS = 100
    for epoch in range(EPOCHS):
        torch.manual_seed(123)
        model.train()
        classifier.train()
        for graph in loader:
            
            optimizer.zero_grad()
            graph = graph.to(device)
    
            out = model(graph).to(device)   
            out = classifier(classifier.embedEdges(out,graph))         
    
            loss = F.nll_loss(out, graph.y, weight=classWeights)
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            print(f'epoch{epoch+1}/{EPOCHS}, loss={loss.item():.4f}')
    
    print("Successfully finished training")    
    print("Saving model to disk")
    with open('data/MyModel.pkl', 'wb') as file: pickle.dump(model, file)
    with open('data/MyModelClass.pkl', 'wb') as file: pickle.dump(classifier, file)
    print("DONE")
    print("------------------- \n") 

print("STARING EVALUATION")  
print("-------------------")
model.eval()
classifier.eval()
try:
    with open('data/OptGreedDiffDataPaths.pkl', 'rb') as file:
        matchCriteriaData = pickle.load(file)
except Exception: matchCriteriaData = dict()
filepaths = []
for key in matchCriteriaData:
    filepaths.append(key)
    print("ADDING DATASET: ", key)
    
filepaths = ["data/Pajek/GD98_b/GD98_b.mtx"]

#for graphId, filepath in enumerate(filepaths):
#    graphId = 0
#    val_original_graphs, val_data, val_y = dl.LoadValGoodCase([filepath])

#val_original_graphs, val_y = dl.WeightsExperiment(val_original_graphs[0])
for graphId, filepath in enumerate(val_original_graphs): #range(1,3):
    
    print("-------------------")
    print("EVALUATING: ", graphId)#, filepath)    


    val_original = val_original_graphs[graphId].to(device)
    print("WEIGHTS: ", val_original.edge_weight[:5])
    val_original_copy = val_original_graphs[graphId].to(device)
    #val_graph = val_data[graphId].to(device)
    
    print("TOTAL NODES ORIGINAL = ", val_original.num_nodes)
    print("TOTAL EDGES ORIGINAL = ", val_original.num_edges)
    
    print("GNN-score based greedy pick")
    
    graph = val_original_copy
    graph, weightSum = MyGCN.GNNMatching(model, classifier, graph)
        
    
    print("Finishing remaining matching with normal greedy.")
    #weightRes, pickedEdgeIndeces = gp.GreedyMatchingLine(val_graph)
    weightRes, pickedEdgeIndeces = gp.GreedyMatchingOrig(graph)
    weightSum += weightRes
    print("Total original edges picked = ", len(pickedEdgeIndeces))
    print("Current weight sum = ", weightSum)
    print("No more matching posibilities left.")
    print("------------------- \n") 
    
    # print("STARTING STANDARD GREEDY SEARCH.")
    # print("-------------------")
    weightGreedy, pickedEdgeIndeces = gp.GreedyMatchingOrig(val_original)
    # print("------------------- \n") 
    
    # print("STARTING RANDOM MATCHING.")
    # print("-------------------")
    weightRandom, pickedEdgeIndeces = gp.RandomMatching(val_original)
    # print("------------------- \n") 
    
    # print("FINAL STATISTICS")
    # print("-------------------")
    true_edges_idx = (val_y[graphId] == 1).nonzero(as_tuple=True)[0]
    true_edges = []
    
    weightMax = 0
    for idx in true_edges_idx:
        from_node = val_original.edge_index[0][idx].item()
        to_node = val_original.edge_index[1][idx].item()
        weightMax += val_original.edge_weight[idx][0]
        true_edges.append((to_node, from_node))     
    opt_matches = len(true_edges_idx)
    opt_drops = len(val_original.edge_attr) - len(true_edges_idx)
    #print("Optimal amount of matches = ", opt_matches)

    weightSum = weightSum.item()
    weightMax = weightMax.item()

    print(f'Total WEIGHT out of optimal: {weightSum:.2f}/{weightMax:.2f} ({100*(weightSum/weightMax):.1f}%) ')
    print(f'Total WEIGHT out of standard greedy: {weightSum:.2f}/{weightGreedy:.2f} ({100*(weightSum/weightGreedy):.1f}%)')
    print(f'Total WEIGHT out of random matching: {weightSum:.2f}/{weightRandom:.2f} ({100*(weightSum/weightRandom):.1f}%)')
    print("FINISHED")
    print("-------------------")    
