import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import GreedyPicker as gp
import MyDataLoader as dl
import ReductionsManager as rm
import MyGCN
import copy
import pickle
import random
from torch.nn.functional import normalize
random.seed(123)
torch.manual_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current CUDA version: ", torch.version.cuda, "\n")


graphs, converted_dataset, target = dl.LoadTrain(limit=10, skipLine=False)#dl.LoadDataCustom()#limit=10)

#graphs2, lineGraphs, target2 = dl.LoadTrain(limit=1000)
#graphs = graphs + graphs2
#target = target + target2
#converted_dataset = converted_dataset + lineGraphs2

# print("Data stats")
# print("----------------")
# print(graphs[0])
largest = 0
largestEdges = 0
for g in graphs:
    if g.num_nodes > largest: largest = g.num_nodes 
    if g.num_edges > largestEdges : largestEdges = g.num_edges

print(f'Largest geraph = {(largest, largestEdges)}')    
print(f'Total graphs = {len(graphs)}')
#print(f'Nodes in graphs graph 0 = {graphs[0].num_nodes}, {len(graphs[0].x)}')
#print(f'Edges in graphs graph 0 = {graphs[0].num_edges}, {len(graphs[0].edge_index[0])}')
# print("-------------------\n")


   
#print("Saveing visualization file")
#dl.VisualizeConverted(val_lineGraphs[0])
#dl.VisualizeOriginal(val_graphs[0])
#print("------------------- \n")

print("STARING TRAINING")
print("-------------------")

torch.manual_seed(123)

#model = MyGCN.MyGCNEdge().to(device)
#classifier = MyGCN.EdgeClassifier().to(device)

model = MyGCN.MyGCN().to(device)


modelLoaded = False
try:
    #raise Exception("TRAINING GREEDY BASED")
    with open('data/tempMyModel.pkl', 'rb') as file:
    #with open('data/MyModel.pkl', 'rb') as file:
        print("Saved model found, skipping training")
        model = pickle.load(file)
        
    with open('data/tempMyModelClass.pkl', 'rb') as file:
    #with open('data/MyModelClass.pkl', 'rb') as file:
        classifier = pickle.load(file)      
        modelLoaded = False
except Exception: print("No model found, training new model")

torch.manual_seed(123)

loader = DataLoader(converted_dataset, batch_size=1, shuffle=True)
#loader = DataLoader(graphs, batch_size=1, shuffle=False)
#optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.0005)#, weight_decay=0.00008)
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.0005)
classWeights = torch.FloatTensor([0.1,0.9]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=classWeights)

if not modelLoaded:
    EPOCHS = 1
    model.train()
    #classifier.train()
    for epoch in range(EPOCHS):
        
        for graph in loader:

            optimizer.zero_grad()
            graph = graph.to(device)
            out = model(graph).to(device)   
            #out = classifier(classifier.embedEdges(out,graph))
            #print(torch.exp(out)[:10])
            #print(graph.y[:10])
            
            loss = F.nll_loss(out, graph.y, weight=classWeights)

            loss.backward()
            optimizer.step()
            
        print(f'epoch{epoch+1}/{EPOCHS}, loss={loss.item():.4f}')
        if epoch%5==0:
            print("Saving temp model to disk")
            with open('data/tempMyModel.pkl', 'wb') as file: pickle.dump(model, file)
            #with open('data/tempMyModelClass.pkl', 'wb') as file: pickle.dump(classifier, file)
            print("DONE")
    print("Successfully finished training")    
    print("Saving model to disk")
    model.eval()
    #classifier.eval()
    with open('data/MyModel.pkl', 'wb') as file: pickle.dump(model, file)
    #with open('data/MyModelClass.pkl', 'wb') as file: pickle.dump(classifier, file)
    print("DONE")
    print("------------------- \n") 

print("STARING EVALUATION")  
print("-------------------")
model.eval()
#classifier.eval()

# try:
#     with open('data/OptGreedDiffDataPaths.pkl', 'rb') as file:
#         matchCriteriaData = pickle.load(file)
# except Exception: matchCriteriaData = dict()

    
val_graphs, val_lineGraphs, val_y = dl.LoadVal(limit=1, skipLine=False)
for graphId, filepath in enumerate(val_graphs): #range(1,3):
    
    print("-------------------")
    print("EVALUATING: ", graphId)#, filepath)    

    val_original = val_graphs[graphId].to(device)
    val_line = val_lineGraphs[graphId].to(device)
    #val_original.edge_weight = normalize(val_original.edge_weight, p=1.0, dim = 0)
    print("WEIGHTS: ", val_original.edge_weight[:5])
    val_original_copy = copy.deepcopy(val_original)
    val_line_copy = copy.deepcopy(val_line)
    
    print("TOTAL NODES ORIGINAL = ", val_original.num_nodes, len(val_original.x))
    print("TOTAL EDGES ORIGINAL = ", val_original.num_edges, len(val_original.edge_index[0]))
    
    print("GNN-score based greedy pick")
    
    graph = val_original_copy
    line = val_line_copy
    #graph, weightSum, ignore, reps = MyGCN.GNNMatching(model, classifier, graph)
    line, weightSum, ignore, reps = MyGCN.GNNMatchingLine(model, line, graph)
        
    print("Finishing remaining matching with normal greedy.")
    weightRes, pickedEdgeIndeces = gp.GreedyMatchingLine(line)
    #weightRes, pickedEdgeIndeces = gp.GreedyMatchingOrig(graph)
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
    true_edges_idx = (val_original.y == 1).nonzero(as_tuple=True)[0]
    true_edges_idx = (val_line.y == 1).nonzero(as_tuple=True)[0]
    
    weightMax, duplicates = 0, set()
    for idx in true_edges_idx:
        
        from_node = val_original.edge_index[0][idx].item()
        to_node = val_original.edge_index[1][idx].item()
        
        if (from_node,to_node) in duplicates or (to_node,from_node) in duplicates: continue
        
        duplicates.add((from_node,to_node))
        duplicates.add((to_node,from_node))
        weightMax += val_original.edge_weight[idx][0]
        
opt_matches = len(true_edges_idx)
opt_drops = len(val_original.edge_attr) - len(true_edges_idx)
#print("Optimal amount of matches = ", opt_matches)
print("Optimal amount of matches = ", opt_matches)

weightSum = weightSum#.item()
weightMax = weightMax.item()

print(f'Total WEIGHT out of optimal: {weightSum:.2f}/{weightMax:.2f} ({100*(weightSum/weightMax):.1f}%) ')
print(f'Total WEIGHT out of standard greedy: {weightSum:.2f}/{weightGreedy:.2f} ({100*(weightSum/weightGreedy):.1f}%)')
print(f'Total WEIGHT out of random matching: {weightSum:.2f}/{weightRandom:.2f} ({100*(weightSum/weightRandom):.1f}%)')
print(f'Greedy remainder after GNN: {weightRes:.2f}/{weightSum:.2f} ({100*(weightRes/weightSum):.1f}%)')
print("FINISHED")
print("-------------------")   

