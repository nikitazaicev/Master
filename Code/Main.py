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

#graphs, lineGraphs, target = dl.LoadTrain(limit=10,doNormalize=True)
#graphs = dl.OverrideToGreedyBased(graphs)
graphs, converted_dataset, target = dl.LoadDataCustom()#limit=10)

# print("Data stats")
# print("----------------")
# print(graphs[0])
# print(f'Total graphs = {len(graphs)}')
# print(f'Nodes in graphs graph 0 = {graphs[0].num_nodes}, {len(graphs[0].x)}')
# print(f'Edges in graphs graph 0 = {graphs[0].num_edges}, {len(graphs[0].edge_index[0])}')
# print("-------------------\n")

val_graphs, val_lineGraphs, val_y = dl.LoadVal(limit=1, doNormalize=True)
   
#print("Saveing visualization file")
#dl.VisualizeConverted(val_lineGraphs[0])
#dl.VisualizeOriginal(val_graphs[0])
#print("------------------- \n")

print("STARING TRAINING")
print("-------------------")

torch.manual_seed(123)

model = MyGCN.MyGCNEdge().to(device)
#model = MyGCN.MyGCN().to(device)
classifier = MyGCN.EdgeClassifier().to(device)


modelLoaded = False
try:
    #raise Exception("TRAINING GREEDY BASED")
    #with open('data/GreedyBased.pkl', 'rb') as file:
    with open('data/MyModel.pkl', 'rb') as file:
        print("Saved model found, skipping training")
        model = pickle.load(file)
        
    #with open('data/GreedyBasedClass.pkl', 'rb') as file:
    with open('data/MyModelClass.pkl', 'rb') as file:
        classifier = pickle.load(file)      
        modelLoaded = True
except Exception: print("No model found, training new model")

torch.manual_seed(123)

#loader = DataLoader(lineGraphs, batch_size=1, shuffle=True)
loader = DataLoader(graphs, batch_size=30, shuffle=False)
optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=0.0005)#, weight_decay=0.0001)
classWeights = torch.FloatTensor([0.1,0.9]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=classWeights)

if not modelLoaded:
    EPOCHS = 50
    for epoch in range(EPOCHS):
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
#    val_graphs, val_lineGraphs, val_y = dl.LoadValGoodCase([filepath])

#val_graphs, val_y = dl.WeightsExperiment(val_graphs[0])
for graphId, filepath in enumerate(val_graphs): #range(1,3):
    
    print("-------------------")
    print("EVALUATING: ", graphId)#, filepath)    

    val_original = val_graphs[graphId].to(device)
    #val_original.edge_weight = normalize(val_original.edge_weight, p=1.0, dim = 0)
    print("WEIGHTS: ", val_original.edge_weight[:5])
    val_original_copy = copy.deepcopy(val_original)
    
    print("TOTAL NODES ORIGINAL = ", val_original.num_nodes, len(val_original.x))
    print("TOTAL EDGES ORIGINAL = ", val_original.num_edges, len(val_original.edge_index[0]))
    
    print("GNN-score based greedy pick")
    
    graph = val_original_copy
    graph, weightSum, ignore = MyGCN.GNNMatching(model, classifier, graph)
        
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
    true_edges_idx = (val_original.y == 1).nonzero(as_tuple=True)[0]
    
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

