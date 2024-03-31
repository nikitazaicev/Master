import matplotlib.pyplot as plt
import numpy as np


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

plt.show()
exit()

# fig, ax = plt.subplots()
# labels = ["default", "degree", "relative difference", "weight difference", "sum" ]
# oneByOneEffect = [0.0, 0.15, 0.14, 0.05, 0.12] # 0.0, 0.13, 0.18, 0.14, 0.20
# init = [0.55, 0.55, 0.55, 0.55, 0.55] # 0.65, 0.78, 0.83, 0.79, 0.85
# additiveEffect = [0.55, 0.70, 0.85, 0.91, 0.92] # 0.65, 0.78, 0.90, 0.89, 0.89

# bar = ax.bar(labels, init, color='white', edgecolor = "black")
# ax.bar_label(bar, label_type='center', fmt='%.2f')
# bar = ax.bar(labels, oneByOneEffect, bottom = init, color='orange', edgecolor = "black")
# ax.bar_label(bar, label_type='center', fmt='%.2f')
# ax.plot(additiveEffect, color='green')
# ax.set_title('Feature augmentation effect')
# ax.set_ylim(bottom=0.0, top=1.2)
# plt.show()

# exit()

#Example data
graphs = ['MNIST trained', 'CUSTOM trained']
#graphs = ['Edge prediction', 'Line graph']
y_pos = np.arange(len(graphs))


#bar_colors = ['tab:blue', 'tab:orange']
#performance = [0.92, 0.88]
performance2 = {

    'GNN': [(1-0.04)*1.01, (1-0.0)*0.79],
    'Remainder': [0.04*1.01, 0.0*0.79]    
    #'GNN': [(1-0.04)*0.73, (1-0.50)*0.85],
    #'Remainder': [0.04*0.73, 0.50*0.85]
    }

fig, ax = plt.subplots()
#hbars = ax.barh(y_pos, [0.92, 0.88], align='center', color=bar_colors)

l = [0]*1

for source, perf in performance2.items():
    print(perf)
    hbars = ax.barh(y_pos, perf, height=0.5, left=l, align='center', label=source)
    l = perf
    ax.bar_label(hbars, label_type='center', fmt='%.2f')

ax.set_yticks(y_pos, labels=graphs)
ax.invert_yaxis()
ax.set_title('Performance comparison on MNIST')
ax.set_xlabel('Performance compared to greedy')
ax.legend(title='Contributor')
# Label with specially formatted floats
#ax.bar_label(hbars, fmt='%.2f')
ax.set_xlim(left=0.5, right=1.3)  # adjust xlim to fit labels
ax.set_ylim(bottom=-0.5, top=1.5)  # adjust xlim to fit labels

plt.show()
exit()



OptAVGTime =  230
GNNAVGTime =  78
GREEDAVGTime =  14
# #----------------
OptAVGWeight =  4637
GNNAVGWeight =  3917
GREEDAVGWeight =  4623
# #----------
remainder =  0.5
# #-----------

fig, ax = plt.subplots(2)
labels = ["GNN", "GREED", "OPT"]
w = [GNNAVGWeight, GREEDAVGWeight, OptAVGWeight]
bar = ax[0].bar(labels, w, color='grey')
ax[0].axhline(y=OptAVGWeight,linewidth=1, color='k')
ax[0].set_ylabel('Weight')
ax[0].set_ylim(bottom=0, top=4800)
ax[0].bar_label(bar, label_type='center', fmt='%.3f')

t = [GNNAVGTime, GREEDAVGTime, OptAVGTime]
ax[1].set_ylabel('Time')
bar = ax[1].bar(labels, t, color='orange', edgecolor = "black")
ax[1].bar_label(bar, label_type='center', fmt='%.0f')
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
    ax.bar_label(hbars, label_type='center', fmt='%.3f')

ax.set_yticks(y_pos, labels=graphs)
ax.invert_yaxis()
ax.set_xlabel('Performance compared to greedy on graph cage8')
ax.set_title('(pick / drop) GNN certainty threshold ')
ax.legend(title='Contributor')
# Label with specially formatted floats
#ax.bar_label(hbars, fmt='%.3f')
ax.set_xlim(left=0.0, right=1.4)  # adjust xlim to fit labels

exit()
