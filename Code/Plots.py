import matplotlib.pyplot as plt
import numpy as np


cage8perf = 0.95 # 0.43 remainder 1000 nodes 10 000 edges
cage9perf = 0.94 # 0.46 remainder 3500 nodes 38 000 edges
cage10perf = 0.95 # 0.55 remainder 11000 nodes 100 000 edges
cage11perf = 0.95 # 0.65 remainder 39000 nodes 500 000 edges

# non reduced
labels = ["cage8","cage9","cage10","cage11"]
nodes_edges = ["1 000 / 10 000","3 500 / 38 000","11 000 / 100 000","39 000 / 500 000"]
remainder = [0.46,0.46, 0.55, 0.65]
perf = [0.95,0.94,1.01,0.95]

# reduced
labels = ["cage8","cage9","cage10","cage11"]
nodes_edges = ["1 000 / 10 000","3 500 / 38 000","11 000 / 100 000","39 000 / 500 000"]
remainder = [0.66,0.63, 0.55, 0.75]
perf = [0.99,0.99,1.01,0.99]

exit()

# # CUSTOM trained performance on sage11
OptAVGTime =  2977.485773086548
GNNAVGTime =  141.2002420425415
GREEDAVGTime =  6.254638910293579
OptAVGWeight =  0.36333611607551575
GNNAVGWeight =  0.34701114892959595
GREEDAVGWeight =  0.36219949417068165
weightResDif =  0.6565099954605103


# #"MNIST trained performance on sage 10"
# OptAVGTime =  238.8848557472229
# GNNAVGTime =  35.6191520690918
# GREEDAVGTime =  0.8406157493591309
# #----------------
# OptAVGWeight =  0.3864327073097229
# GNNAVGWeight =  0.3369670510292053
# GREEDAVGWeight =  0.38539551471512823
# #----------------
# remainder =  0.5323471426963806

# # CUSTOM trained performance on sage10
# OptAVGTime =  230.7519097328186
# GNNAVGTime =  36.04787516593933
# GREEDAVGTime =  0.709648847579956
# #----------------
# OptAVGWeight =  0.3864327073097229
# GNNAVGWeight =  0.36708855628967285
# GREEDAVGWeight =  0.38539551471512823
# #----------
# remainder =  0.5425851345062256
# #-----------

# fig, ax = plt.subplots(2)
# labels = ["GNN", "GREED", "OPT"]
# w = [GNNAVGWeight, GREEDAVGWeight, OptAVGWeight]
# bar = ax[0].bar(labels, w, color='grey')
# ax[0].axhline(y=OptAVGWeight,linewidth=1, color='k')
# ax[0].set_ylabel('Weight')
# ax[0].set_ylim(bottom=0.3, top=0.4)
# ax[0].bar_label(bar, label_type='center', fmt='%.3f')

# t = [GNNAVGTime, GREEDAVGTime, OptAVGTime]
# ax[1].set_ylabel('Time')
# bar = ax[1].bar(labels, t, color='orange', edgecolor = "black")
# ax[1].bar_label(bar, label_type='center', fmt='%.0f')
# plt.show()

# exit()
# fig, ax = plt.subplots()
# labels = ["default", "degree", "weight difference", "sum", "relative difference" ]
# oneByOneEffect = [0.0, 0.15, 0.14, 0.05, 0.12]
# init = [0.55, 0.55, 0.55, 0.55, 0.55]
# additiveEffect = [0.55, 0.70, 0.85, 0.91, 0.92]

# bar = ax.bar(labels, init, color='white', edgecolor = "black")
# ax.bar_label(bar, label_type='center', fmt='%.2f')
# bar = ax.bar(labels, oneByOneEffect, bottom = init, color='orange', edgecolor = "black")
# ax.bar_label(bar, label_type='center', fmt='%.2f')
# ax.plot(additiveEffect, color='green')
# ax.set_title('Feature augmentation effect')
# ax.set_ylim(bottom=0.0, top=1.2)
# plt.show()

# exit()

graphs = ('50% / 0%', '50% / 10%', '50% / 50%', '60% / 0%', '60% / 60%', '70% / 0%', '70% / 70%')
performance = [0.35/0.385, 0.17/0.385, 0.16/0.385, 0.381/0.385, 0.13/0.385, 0.3856/0.3853, 0.12/0.385]

performance2 = {
    'GNN': [(1-0.54)*0.367/0.385, 
            (1-0.0)*0.17/0.385, 
            (1-0.0)*0.16/0.385, 
            (1-0.65)*0.381/0.385, 
            (1-0.0)*0.13/0.385,
            (1-0.66)*0.3856/0.3853, 
            (1-0.0)*0.12/0.385],
    'Remainder': 
            [(0.54)*0.367/0.385, 
            (0.0)*0.17/0.385, 
            (0.0)*0.16/0.385, 
            (0.65)*0.381/0.385, 
            (0.0)*0.13/0.385,
            (0.66)*0.3856/0.3853, 
            (0.0)*0.12/0.385]
    }
y_pos = np.arange(len(graphs))

fig, ax = plt.subplots()

l = [0]*7
sums = [0]*7

for source, perf in performance2.items():
    hbars = ax.barh(y_pos, perf, left=l, align='center', label=source)
    l = perf
    ax.bar_label(hbars, label_type='center', fmt='%.3f')

ax.set_yticks(y_pos, labels=graphs)
ax.invert_yaxis()
ax.set_xlabel('Performance compared to greedy on graph sage10')
ax.set_title('(pick / drop) GNN certainty threshold ')
ax.legend(title='Contributor')
# Label with specially formatted floats
#ax.bar_label(hbars, fmt='%.3f')
ax.set_xlim(left=0.0, right=1.4)  # adjust xlim to fit labels

exit()
# Example data
graphs = ('Line Graph', 'Standard')
y_pos = np.arange(len(graphs))


#bar_colors = ['tab:blue', 'tab:orange']
#performance = [0.92, 0.88]
performance2 = {
    
    'GNN': [(1-0.03)*0.92, (1-0.04)*0.88],
    'Remainder': [0.03*0.92, 0.04*0.88]
    }

fig, ax = plt.subplots()
#hbars = ax.barh(y_pos, [0.92, 0.88], align='center', color=bar_colors)

l = [0]*2

for source, perf in performance2.items():
    hbars = ax.barh(y_pos, perf, left=l, align='center', label=source)
    l = perf
    ax.bar_label(hbars, label_type='center', fmt='%.2f')

ax.set_yticks(y_pos, labels=graphs)
ax.invert_yaxis()
ax.set_title('MNIST dataset test')
ax.set_xlabel('Performance compared to greedy')
ax.legend(title='Contributor')
# Label with specially formatted floats
#ax.bar_label(hbars, fmt='%.2f')
ax.set_xlim(left=0.5, right=1.2)  # adjust xlim to fit labels

plt.show()