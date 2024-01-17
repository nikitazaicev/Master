import pickle
import MyGCN
import DataLoader as dl
import torch
import random
import GreedyPicker as gp
import time
import LineGraphConverter as lgc
import ReductionsManager as rm

torch.manual_seed(123)
random.seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current CUDA version: ", torch.version.cuda, "\n")

try:
    with open('data/MyModel.pkl', 'rb') as file:
        print("Loading Model")
        model = pickle.load(file).to(device)
    with open('data/MyModelClass.pkl', 'rb') as file:
        classifier = pickle.load(file).to(device) 
        modelLoaded = True
except Exception: print("No model found. EXIT")


graphs, converted_dataset, target = dl.LoadTest(limit=10)
print("----------------")

resultsGNN = []
resultsGreedy = []

for idx, graph in enumerate(graphs):
    
    weightSum = 0
    graph, weightRed = rm.ApplyReductionRules(graph)
    weightSum += weightRed
    start_time = time.time()
    weightGreedy, pickedEdgeIndeces = gp.GreedyMatchingOrig(graph)
    timeTotal = time.time() - start_time
    resultsGreedy.append([("TIME", timeTotal),("WEIGHT", weightGreedy)])
    
    graph = graph.to(device)
    start_time = time.time()
    graph.x = lgc.AugmentOriginalNodeFeatures(graph)
    graph, weightGnn = MyGCN.GNNMatching(model, classifier, graph.to(device))
    weightRes, pickedEdgeIndeces = gp.GreedyMatchingOrig(graph)
    weightSum += weightGnn
    weightSum += weightRes
    timeTotal = time.time() - start_time
    resultsGNN.append([("TIME", timeTotal),("WEIGHT", weightSum.item())])
    
avgTimeGnn, avgWeightGnn, avgTimeGreed, avgWeightGreed = 0,0,0,0
for idx, r in enumerate(resultsGNN):
    #print("GNN = ", resultsGNN[idx])
    #print("GREED = ", resultsGreedy[idx])
    #print("----------------")
    avgTimeGnn += resultsGNN[idx][0][1]
    avgWeightGnn += resultsGNN[idx][1][1]
    avgTimeGreed += resultsGreedy[idx][0][1]
    avgWeightGreed += resultsGreedy[idx][1][1]
    
    
print("GNN AVG Time = ", avgTimeGnn/len(resultsGNN))
print("GREED AVG Time = ", avgTimeGreed/len(resultsGNN))

print("GNN AVG Weight = ", avgWeightGnn/len(resultsGNN))
print("GREED AVG Weight = ", avgWeightGreed/len(resultsGNN))

print("----------------")















