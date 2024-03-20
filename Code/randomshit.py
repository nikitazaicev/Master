import torch
from blossom import maxWeightMatching
import pickle
import MyGCN
import MyDataLoader as dl
import torch
import random
import GreedyPicker as gp
import time
from blossom import maxWeightMatching
import LineGraphConverter as lgc
import ReductionsManager as rm
import ssgetpy as ss
from torch_geometric.data import Data
from scipy.io import mmread
import MyGCN
torch.manual_seed(123)
random.seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current CUDA version: ", torch.version.cuda, "\n")
print("TEST BIG GRAPH")



try:
    #with open('data/MNISTTRAINED/MyModel.pkl', 'rb') as file:
    #with open('data/CUSTOMTRAINED/MyModel.pkl', 'rb') as file:
    with open('data/tempMyModel.pkl', 'rb') as file:
        print("Loading Model")
        model = pickle.load(file).to(device)
        model.eval()
    #with open('data/MNISTTRAINED/MyModelClass.pkl', 'rb') as file:
    #with open('data/CUSTOMTRAINED/MyModelClass.pkl', 'rb') as file:
    with open('data/tempMyModelClass.pkl', 'rb') as file:
        classifier = pickle.load(file).to(device) 
        classifier.eval()
        modelLoaded = True
except Exception: print("No model found. EXIT")


badGreedyGraph = Data()
badGreedyGraph.num_nodes = 4
badGreedyGraph.num_edges = 6
badGreedyGraph.edge_index = torch.LongTensor([[0,1,1,2,2,3],
                                              [1,0,2,1,3,2]]).to(device)
badGreedyGraph.edge_weight =  torch.FloatTensor([[0.5],[0.5],[0.9],[0.9],[0.5],[0.5]]).to(device) 
badGreedyGraph.edge_attr = badGreedyGraph.edge_weight.flatten().to(device)
badGreedyGraph.x, badGreedyGraph.adj = lgc.AugmentOriginalNodeFeatures(badGreedyGraph)
badGreedyGraph = badGreedyGraph.to(device)

weightGreedy, pickedEdgeIndeces = gp.GreedyMatchingOrig(badGreedyGraph)
print(weightGreedy)

weightSum = 0
badGreedyGraph, weightGnn, ignore, reps = MyGCN.GNNMatching(model, classifier, badGreedyGraph, 0.5, 0.0, verbose=False)
weightRes, pickedEdgeIndeces = gp.GreedyMatchingOrig(badGreedyGraph)

weightSum += weightGnn
weightSum += weightRes
weightResDif = weightRes/weightSum
print(weightGnn, reps)
exit()

import MyDataLoader as dl
from scipy.io import mmread
file_name = 'data/customtrain/converted_dataset.pkl'
with open(file_name, 'rb') as file:
    print(file_name, " loaded")
    convertedset = pickle.load(file)
    
print(len(convertedset))    
workingfuckingfiles = []
file_name = 'data/custom/customdatasetfiles.pkl'
with open(file_name, 'rb') as file:
    print(file_name, " loaded")
    filenames = pickle.load(file)
print("LOADING AND PROCCESSING DATA")
data = []
target = []
count = 0
for filename in filenames:
    # print("Reading ", filename)
    # mmformat = mmread(f'data/custom/{filename}/{filename}.mtx').toarray()
    # graph = dl.FromMMformat(mmformat)
    
    # data.append(graph)
    # file_name = f'data/customtrain/weights/{filename}.pkl'
    # with open(file_name, 'wb') as file:
    #     pickle.dump(graph.edge_weight, file)
    #     print(f'Object successfully saved to weights/"{file_name}"')
    # count += 1
    # if (count) % 10 == 0: print(f'graph{count}/{len(filenames)}')
    graph = badGreedyGraph
    print("Blossom matching and line graph convertion")

    blossominput, uniqueEdges = [], set()
    for idx2 in range(len(graph.edge_index[0])):
        (f,t) = (graph.edge_index[0][idx2].item(), graph.edge_index[1][idx2].item())
        w = graph.edge_attr[idx2].item()
        if (f,t) not in uniqueEdges and (t,f) not in uniqueEdges:
            blossominput.append((f,t,w))
            uniqueEdges.add((f,t))
            uniqueEdges.add((t,f))
    try:
        targetClasses = dl.AssignTargetClasses(graph, maxWeightMatching(blossominput))
    except: 
        print("blossom failed skipping")
        continue
    workingfuckingfiles.append(filename)
    target.append(targetClasses)
print(len(filenames))
print(len(workingfuckingfiles))

file_name = 'data/custom/customdatasetfiles.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(workingfuckingfiles, file)
    print(f'Object successfully saved to "{file_name}"')

file_name = 'data/customtrain/target_data.pkl'
with open(file_name, 'wb') as file:
    pickle.dump(target, file)
    print(f'Object successfully saved to "{file_name}"')
    
file_name = 'data/customtrain/converted_dataset.pkl'
with open(file_name, 'wb') as file:
    pickle.dump([], file)
    print(f'Object successfully saved to "{file_name}"')
