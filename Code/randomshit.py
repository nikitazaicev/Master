import torch
from blossom import maxWeightMatching
import pickle

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
    print("Reading ", filename)
    mmformat = mmread(f'data/custom/{filename}/{filename}.mtx').toarray()
    graph = dl.FromMMformat(mmformat)
    
    data.append(graph)
    file_name = f'data/customtrain/weights/{filename}.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(graph.edge_weight, file)
        print(f'Object successfully saved to weights/"{file_name}"')
    count += 1
    if (count) % 10 == 0: print(f'graph{count}/{len(filenames)}')
    
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
