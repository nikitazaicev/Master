import pickle
import MyGCN
import MyDataLoader as dl
import torch
import random
import GreedyPicker as gp
import time
import LineGraphConverter as lgc
import ReductionsManager as rm
from torch.nn.functional import normalize
from scipy.io import mmread

torch.manual_seed(123)
random.seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current CUDA version: ", torch.version.cuda, "\n")

try:
    #with open('data/MNISTTRAINED/MyModel.pkl', 'rb') as file:
    #with open('data/CUSTOMTRAINED/MyModel.pkl', 'rb') as file:
    with open('data/tempMyModel.pkl', 'rb') as file:
        print("Loading Model")
        model = pickle.load(file).to(device)
    #with open('data/MNISTTRAINED/MyModelClass.pkl', 'rb') as file:
    #with open('data/CUSTOMTRAINED/MyModelClass.pkl', 'rb') as file:
    with open('data/tempMyModelClass.pkl', 'rb') as file:
        classifier = pickle.load(file).to(device) 
        modelLoaded = True
except Exception: print("No model found. EXIT")
model.eval()
classifier.eval()

graphs, converted_dataset, target = dl.LoadVal(limit=100)

#graphs, converted_dataset, target = dl.LoadTrain(limit=9, doNormalize=False)

# with open('data/OptGreedDiffDataPaths.pkl', 'rb') as file:
#     goodcases = pickle.load(file)
# paths = []
# for item in goodcases: 
#     print(item)
#     paths.append(item)
# graphs, converted_dataset, target = dl.LoadValGoodCase(paths[1:])

#graphs, converted_dataset, target = dl.LoadDataCustom(limit=10)

# graphs, filename = [], "data/Pajek/GD98b/GD98b.mtx"
# print("Reading ", filename)
# mmformat = mmread(f'data/custom/GD98_b/GD98_b.mtx').toarray()

# graph = dl.FromMMformat(mmformat)
# randWeights = torch.rand(((int(len(graph.edge_attr)/2))))
# graph.edge_attr = torch.cat((randWeights, randWeights))
# graph.edge_weight = torch.reshape(graph.edge_attr,((len(graph.edge_attr),1)))
# graphs.append(graph)
# graphs, converted, target = dl.ProccessData(graphs , "custom", skipLine=True)  

print("Total graphs = ", len(graphs))
print("----------------")

resultsGNN, resultsGreedy, resultsOpt = [], [], []
useReduction = False

for idx, graph in enumerate(graphs):
    
    weightSum, weightRed = 0, 0
    if useReduction:
        print("Before reduction = ", graph.num_nodes)
        graph, weightRed = rm.ApplyReductionRules(graph)
        weightSum += weightRed
        print("After reduction = ", graph.num_nodes, weightRed)
    
    true_edges_idx = (graph.y == 1).nonzero(as_tuple=True)[0]    
    
    totalWeightOpt, duplicates = 0, set()
    start_time = time.time()
    for idx in true_edges_idx:
        from_node = graph.edge_index[0][idx].item()
        to_node = graph.edge_index[1][idx].item()
        if (from_node,to_node) in duplicates or (to_node,from_node) in duplicates: continue
        duplicates.add((from_node,to_node))
        duplicates.add((to_node,from_node))
        #totalWeightOpt += graph.edge_attr[idx].item() + weightRed  
        totalWeightOpt += graph.edge_weight[idx][0].item() + weightRed  
    timeTotal = time.time() - start_time
    resultsOpt.append([("TIME", timeTotal),("WEIGHT", totalWeightOpt)])
        
    start_time = time.time()
    weightGreedy, pickedEdgeIndeces = gp.GreedyMatchingOrig(graph)

    weightGreedy += weightRed 
    timeTotal = time.time() - start_time
    resultsGreedy.append([("TIME", timeTotal),("WEIGHT", weightGreedy)])
    
    start_time = time.time()
    #graph.x = lgc.AugmentOriginalNodeFeatures(graph)
    graph = graph.to(device)
    graph, weightGnn, ignore, steps = MyGCN.GNNMatching(model, classifier, graph, 0.5, 0.0, verbose=False)
    print("gnn steps ", steps )
    weightRes, pickedEdgeIndeces = gp.GreedyMatchingOrig(graph)

    weightSum += weightGnn
    weightSum += weightRes
    weightResDif = weightRes/weightSum

    timeTotal = time.time() - start_time
    resultsGNN.append([("TIME", timeTotal),("WEIGHT", weightSum),("Gnn Res weight Diff", weightResDif)])
    
avgTimeGnn, avgWeightGnn, avgTimeGreed, avgWeightGreed, GnnResDif, weightResDif = 0,0,0,0,0,0
avgTimeOpt, avgWeightOpt = 0,0
for idx, r in enumerate(resultsGNN):
    avgTimeGnn += resultsGNN[idx][0][1]
    avgWeightGnn += resultsGNN[idx][1][1]
    avgTimeGreed += resultsGreedy[idx][0][1]
    avgWeightGreed += resultsGreedy[idx][1][1]
    avgTimeOpt += resultsOpt[idx][0][1]
    avgWeightOpt += resultsOpt[idx][1][1]
    weightResDif += resultsGNN[idx][2][1]


print("-----------------------------------------------")    

print("OPT AVG Time = ", avgTimeOpt/len(resultsGNN))
print("OPT AVG Weight = ", avgWeightOpt/len(resultsGNN))    

print("-----------------------------------------------")    

print("GNN AVG Time = ", avgTimeGnn/len(resultsGNN))
print("GNN AVG Weight = ", (avgWeightGnn/len(resultsGNN)))
   
print("-----------------------------------------------")

print("GREED AVG Time = ", avgTimeGreed/len(resultsGNN))
print("GREED AVG Weight = ", avgWeightGreed/len(resultsGNN))
   
print("-----------------------------------------------")

print("weightResDif = ", (weightResDif/len(resultsGNN)))

print("----------------")















