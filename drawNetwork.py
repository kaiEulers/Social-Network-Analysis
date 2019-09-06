"""
@author: kaisoon
"""
# ----- Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import importlib
import groupActor as group
import colourPals as cp


#%% Get node position in kamada kawai layout
DATE = '2017-12'
pos = nx.kamada_kawai_layout(G)
# Save position coordinates
nx.write_gpickle(pos, f"results/ssm_{DATE}_nodePos_kamadaKawai.gpickle")


#%% Get node position in spring layout
DATE = '2017-12'
# Increase k to spread nodes further apart
pos = nx.spring_layout(G, iterations=300, k=7)
# Save position coordinates
nx.write_gpickle(pos, f"results/ssm_{DATE}_nodePos_spring.gpickle")


#%% ----- Draw Network
importlib.reload(group)
importlib.reload(cp)

# ----- Load graph
DATE = '2017-12'
FILE_NAME = f"ssm_{DATE}_results_NMF_senti.csv"
FILE_NAME_GRAPH = FILE_NAME.replace('NMF_senti.csv', 'graph.png')

results = pd.read_csv(f"results/{FILE_NAME}")
cent = pd.read_csv(f"results/{FILE_NAME.replace('NMF_senti', 'centrality')}")
G = nx.read_gpickle(f"results/{FILE_NAME_GRAPH.replace('.png', '.gpickle')}")
# TODO: Load normal graph with weighted edges

# ----- Preprocessing for node colouring
CENT_THRES = 0.75
# Find actor/s above the centrality threshold
actors_highCent = list(cent[cent['Degree'] > CENT_THRES]['Person'])


# Group by actors by party for node colouring
groupedActors, colourMap, legend = group.byAttr(G, 'Party')
# groupedActors, colourMap, legend = group.byAttr(G, 'Gender')
# groupedActors, colourMap, legend = group.byAttr(G, 'Metro')

# ----- Set up graph layout
FILE_NAME_LAYOUT = f"results/ssm_{DATE}_nodePos_kamadaKawai.gpickle"
# FILE_NAME_LAYOUT = f"results/ssm_{DATE}_nodePos_spring.gpickle"
NODE_SIZE = 10
NODE_ALPHA = 0.75
LABEL_SIZE = 7
FIG_SiZE = 4

sns.set_style("darkgrid")
sns.set_context("notebook")

# Load position of nodes
pos = nx.read_gpickle(FILE_NAME_LAYOUT)

# ----- Draw nodes
# TODO: Write function to colour nodes by single colour (default gray)
# TODO: Write function to colour nodes base on attribute
fig = plt.figure(figsize=(FIG_SiZE*3, FIG_SiZE*2), dpi=300, tight_layout=True)
plt.title(f"Same-sex Marriage Bill {DATE}")

for (grp, leg) in zip(list(groupedActors.keys()), legend):
    nx.draw_networkx_nodes(G, pos, nodelist=groupedActors[grp],
                           node_size=NODE_SIZE*100, node_color=colourMap[grp],
                           alpha=NODE_ALPHA, edgecolors='black', linewidths=0.5, label=leg)
plt.legend()


# ----- Draw node labels
# Construct dict to reformat names from 'first last' to 'first\nlast'
nodes = G.nodes
nodeLabels = []
for n in nodes:
    nodeLabels.append(n.replace(' ', '\n'))
nodeLabels = dict(zip(nodes, nodeLabels))

# Divide nodeLabels into with and without actors with high centrality
nodeLabels_wo_highCent = {k: v for (k, v) in nodeLabels.items() if k not in actors_highCent}
nodeLabels_w_highCent = {k: v for (k, v) in nodeLabels.items() if k in actors_highCent}
# Draw labels
nx.draw_networkx_labels(G, pos, labels=nodeLabels_wo_highCent, font_size=LABEL_SIZE, font_color='black')
nx.draw_networkx_labels(G, pos, labels=nodeLabels_w_highCent, font_size=LABEL_SIZE*2, font_color='black', font_weight='bold')


# ----- Draw edges
# TODO: Write function to colour edges with thickness unweighted and a single colour (default gray)
# TODO: Write function to colour edges with thickness weighted and a single colour (default gray)

nx.draw_networkx_edges(G, pos, width=0.25, edge_color=cp.cbDark2['gray'])


# ----- Save figure
fig.savefig(f"results/{FILE_NAME_GRAPH}", dpi=300)
plt.show()
