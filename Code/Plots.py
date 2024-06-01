import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import pandas as pd



fig, ax = plt.subplots(figsize=(8, 6),dpi=350)
labels = ["default", "degree", "relative difference", "sum", "weight difference",  "All at once" ]
#plt.xticks([i*1 for i in range(6)], labels)
oneByOneEffect = [0.0, 0.13, 0.11, 0.09, 0.02, 0.30] # 0.0, 0.13, 0.18, 0.14, 0.20
init = [0.62, 0.62, 0.62, 0.62, 0.62, 0.62] # 0.65, 0.78, 0.83, 0.79, 0.85
additiveEffect = [0.59, 0.74, 0.85, 0.91, 0.92] # 0.65, 0.78, 0.90, 0.89, 0.89

bar = ax.bar(labels, init, width=0.3,  color='white', edgecolor = "black")
ax.bar_label(bar, label_type='center', fmt='%.2f')
bar = ax.bar(labels, oneByOneEffect, width=0.3, bottom = init, color='green', edgecolor = "black")
ax.bar_label(bar, label=oneByOneEffect, label_type='edge', fmt='%.2f')
#ax.plot(additiveEffect, color='green')
ax.set_title('Feature augmentation')
ax.set_ylim(bottom=0.0, top=1.2)
plt.ylabel("Perormance compared to greedy")
plt.savefig('FeatureAugmentationLine.png', bbox_inches='tight')

plt.show()
exit()
labels = ["G22 \n nodes: 2000 \n edges: 40000",
          "cage9 \n 3500 \n 38000",
          "Power \n 5000 \n 13000",
          "California \n 5000 \n 20000",
          "G55 \n 5000 \n 25000",
          "cage10 \n 11000 \n 140000",
          "as-22july06 \n 22000 \n 96000",
          "dictionary28 \n 39000 \n 178000"]


penguin_means = {
    'GNN MNIST-trained':  [ 36, 50,  115, 113, 127,  34, 109, 102],
    'GNN custom-trained': [101, 95,  137, 140, 93,   97, 110, 117],
    'Greedy':             [100, 100, 100, 100, 100, 100, 100, 100],
    'Blossom':            [190, 184, 141, 160, 165, 188, 120, 138]
}

x = np.arange(len(labels))  # the label locations

width = 0.2  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(14, 5))

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Performance compared to greedy (%)')
ax.set_title('Performance comparison on greedy disadvantaged graphs')
ax.set_xticks(x + 1.5*width, labels)
ax.legend(loc='upper right', ncols=2)
ax.set_ylim(0, 250)
#ax.set_xlim(-0.5, 7)

plt.savefig('MutatedGraphsTest.png', dpi=2200)
plt.show()
exit()

#Example data
graphs = ['Line graph','Edge prediction', 'Blossom']
#graphs = ['Edge prediction', 'Line graph']
y_pos = np.arange(len(graphs))


#bar_colors = ['tab:blue', 'tab:orange']
#performance = [0.92, 0.88]
performance2 = {

    'GNN/Algorithm': [(1-0.04)*1.03,(1-0.04)*1.01, 1.06],
    'Remainder': [0.04*1.03, 0.04*1.01, 0]    
    #'GNN': [(1-0.04)*0.73, (1-0.50)*0.85],
    #'Remainder': [0.04*0.73, 0.50*0.85]
    }

fig, ax = plt.subplots()
#hbars = ax.barh(y_pos, [0.92, 0.88], align='center', color=bar_colors)

l = [0]*1
labeltype='center'
for source, perf in performance2.items():
    print(perf)
    hbars = ax.barh([0,0.4,0.8], perf, height=0.2, left=l, align='center', label=source)
    l = perf
    ax.bar_label(hbars, label_type=labeltype, fmt='%.2f')
    labeltype='edge'

ax.set_yticks([0,0.4,0.8], labels=graphs)
ax.invert_yaxis()
ax.set_title('Performance comparison on MNIST')
ax.set_xlabel('Performance compared to greedy')
ax.legend(title='Contributor')
# Label with specially formatted floats
#ax.bar_label(hbars, fmt='%.2f')
ax.set_xlim(left=0.4, right=1.2)  # adjust xlim to fit labels
ax.set_ylim(bottom=-0.2, top=1.35)  # adjust xlim to fit labels
plt.savefig('LineVSEdgeOnMNIST.png', dpi=2000, bbox_inches='tight')
plt.show()
exit()




OptAVGTime =  230
GNNAVGTime =  71
GREEDAVGTime =  14
# #----------------
OptAVGWeight =  4637
GNNAVGWeight =  3415
GREEDAVGWeight =  4623

OptAVGTime =  230
GNNAVGTime =  78
GREEDAVGTime =  14
# #----------------
OptAVGWeight =  4637
GNNAVGWeight =  3917
GREEDAVGWeight =  4623


fig, ax = plt.subplots(2)
labels = ["Edge classification", "GREEDY", "OPTIMAL"]
w = [GNNAVGWeight, GREEDAVGWeight, OptAVGWeight]
ax[0].axhline(y=OptAVGWeight,linewidth=1, color='k')
ax[0].set_ylabel('Weight')
ax[0].set_ylim(bottom=0, top=4900)
bar1 = ax[0].bar(labels, w, color='grey')
ax[0].bar_label(bar1, label_type='center', fmt='%.1f')


t = [GNNAVGTime, GREEDAVGTime, OptAVGTime]
ax[1].set_ylabel('Time')
bar2 = ax[1].bar(labels, t, color='orange', edgecolor = "black")
ax[1].set_ylim(bottom=0, top=280)
ax[1].bar_label(bar2, label_type='center', fmt='')
for rect in bar2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')


# Add counts above the two bar graphs


plt.savefig('CUSTOMtrainCAGE10', dpi=2000, bbox_inches='tight')
plt.show()
exit()







graphs = ('50% / 0%', '50% / 10%', '50% / 20%', '60% / 0%', '70% / 0%', '80% / 0%', '90% / 0%', '95% / 0%') 

y_pos = np.arange(len(graphs))
performance = [0.837,0.586,0.586,0.871,0.891,0.966,0.992,0.999]
total = 411
performance2 = {
    'GNN': [(1-0.19)*344/total, 
            (1-0.0)*241/total, 
            (1-0.0)*241/total, 
            (1-0.42)*358/total, 
            (1-0.65)*366/total, 
            (1-0.85)*397/total,
            (1-0.91)*408/total,
            (1-0.99)*411/total],
    'Remainder': 
            [(0.19)*344/total, 
            (0.0)*241/total, 
            (0.0)*241/total,
            (0.42)*358/total,
            (0.65)*366/total,
            (0.85)*397/total,
            (0.91)*408/total,
            (0.99)*411/total]
    }
fruit_counts = [4000, 2000, 7000]

fig, ax = plt.subplots(figsize=(10,6))
h=[0]*len(performance)
for source, perf in performance2.items():
    t = [x + y for x, y in zip(h, perf)]
    bars = ax.bar(graphs, perf, bottom=h, align='center', label=source)
    h=perf
    ax.bar_label(bars, padding=5,label_type='center', fmt='%.2f')


ax.set(ylabel='performance compared to greedy', title='(pick/drop) GNN certainty threshold', ylim=(0, 1.1))
ax.legend(title='Contributor')
plt.savefig('ThresholdDemo.png', dpi=2000, bbox_inches='tight')
plt.show()

exit()

labels = ["G22 \n nodes: 2000 \n edges: 40000",
          "cage9 \n 3500 \n 38000",
          "California \n 5000 \n 20000",
          "G55 \n 5000 \n 25000",
          "cage10 \n 11000 \n 140000",
          "as-22july06 \n 22000 \n 96000",
          "dictionary28 \n 39000 \n 178000"]


penguin_means = {
    'Performance': [96,86,91,92,95,87,92],
    'Time': [69,120,66,25,40,20,3],
    'Remainder': [95,57,6,10,85,34,3]
}

x = np.arange(len(labels))  # the label locations

width = 0.15  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(10, 6),dpi=400)

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Relative to greedy (%), except time')
ax.set_title('GNN performance on different graphs')
ax.set_xticks(x + width, labels)
ax.legend(loc='upper right', ncols=3)
ax.set_ylim(0, 130)
#ax.set_xlim(-0.5, 7)

plt.savefig('FINALResults.png')
plt.show()

exit()


OptAVGTime =  230
GNNAVGTime =  78
GREEDAVGTime =  14
# #----------------
OptAVGWeight =  4637
GNNAVGWeight =  3917
GREEDAVGWeight =  4623

OptAVGTime =  230
GNNAVGTime =  71
GREEDAVGTime =  14
# #----------------
OptAVGWeight =  4637
GNNAVGWeight =  3415
GREEDAVGWeight =  4623

fig, ax = plt.subplots(2)
labels = ["GNN", "GREED", "OPT"]
w = [GNNAVGWeight, GREEDAVGWeight, OptAVGWeight]
ax[0].axhline(y=OptAVGWeight,linewidth=1, color='k')
ax[0].set_ylabel('Weight')
ax[0].set_ylim(bottom=0, top=4900)
bar1 = ax[0].bar(labels, w, color='grey')
ax[0].bar_label(bar1, label_type='center', fmt='%.1f')


t = [GNNAVGTime, GREEDAVGTime, OptAVGTime]
ax[1].set_ylabel('Time')
bar2 = ax[1].bar(labels, t, color='orange', edgecolor = "black")
ax[1].set_ylim(bottom=0, top=280)
ax[1].bar_label(bar2, label_type='center', fmt='')
for rect in bar2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.0f}', ha='center', va='bottom')


# Add counts above the two bar graphs


plt.savefig('MNISTtrainCAGE10.png', dpi=2000, bbox_inches='tight')
plt.show()
exit()




labels = ["G22 \n nodes: 2000 \n edges: 40000",
          "cage9 \n 3500 \n 38000",
          "California \n 5000 \n 20000",
          "G55 \n 5000 \n 25000",
          "cage10 \n 11000 \n 140000",
          "as-22july06 \n 22000 \n 96000",
          "dictionary28 \n 39000 \n 178000"]


penguin_means = {
    'Performance': [96,86,91,92,95,87,92],
    'Time': [69,120,66,25,40,20,3],
    'Remainder': [95,57,6,10,85,34,3]
}

x = np.arange(len(labels))  # the label locations

width = 0.15  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(10, 3))

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Relative to greedy (%), except time')
ax.set_title('GNN performance on different graphs')
ax.set_xticks(x + width, labels)
ax.legend(loc='upper right', ncols=3)
ax.set_ylim(0, 130)
#ax.set_xlim(-0.5, 7)

plt.savefig('FINALResults.png', dpi=2000)
plt.show()

exit()





graphs = ('50% / 0%', '50% / 10%', '50% / 20%', '60% / 0%', '70% / 0%', '80% / 0%', '90% / 0%', '95% / 0%') 
total = 411
performance2 = {
    'GNN': [(1-0.19)*344/total, 
            (1-0.0)*241/total, 
            (1-0.0)*241/total, 
            (1-0.42)*358/total, 
            (1-0.65)*366/total, 
            (1-0.85)*397/total,
            (1-0.91)*408/total,
            (1-0.99)*411/total],
    'Remainder': 
            [(0.19)*344/total, 
            (0.0)*241/total, 
            (0.0)*241/total,
            (0.42)*358/total,
            (0.65)*366/total,
            (0.85)*397/total,
            (0.91)*408/total,
            (0.99)*411/total]
    }
y_pos = np.arange(len(graphs))

fig, ax = plt.subplots()

l = [0]*8
sums = [0]*8

for source, perf in performance2.items():
    hbars = ax.barh(y_pos, perf, left=l, align='center', label=source)
    l = perf
    ax.bar_label(hbars, label_type='center', fmt='%.2f')

ax.set_yticks(y_pos, labels=graphs)
ax.invert_yaxis()
ax.set_xlabel('Performance compared to greedy on graph cage8')
ax.set_title('(pick / drop) GNN certainty threshold ')
ax.legend(title='Contributor')
# Label with specially formatted floats
#ax.bar_label(hbars, fmt='%.3f')
ax.set_xlim(left=0.0, right=1.4)  # adjust xlim to fit labels

exit()
