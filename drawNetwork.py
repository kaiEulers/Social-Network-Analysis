#%% ----- Imports
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import re
import importlib
import kai_groupActor as grp
import kai_colourPals as cp


#%% ----- Load graph, centrality, and clique data
FILE_NAME = "ssm_results_NMF_senti_2017-12.csv"
graphFileName = re.sub('.csv', '.gpickle', FILE_NAME)
graphFileName = re.sub('NMF_senti', 'graph', graphFileName)
G = nx.read_gpickle(f"results/{graphFileName}")
# TODO: Load normal graph with weighted edges

cent = pd.read_csv(f"results/{re.sub('NMF_senti', 'centrality', FILE_NAME)}")
cliques = pd.read_csv(f"results/{re.sub('NMF_senti', 'cliques', FILE_NAME)}", header=None, index_col=0)


#%% Get node position in kamada kawai layout
pos = nx.kamada_kawai_layout(G)
# Save position coordinates
nx.write_gpickle(pos, "results/ssm_nodePos_kamadaKawai_2017-12.gpickle")

#%% Get node position in spring layout
# Increase k to spread nodes further apart
pos = nx.spring_layout(G, iterations=300, k=9)
# Save position coordinates
nx.write_gpickle(pos, "results/ssm_nodePos_spring_2017-12.gpickle")


#%% ----- Draw Network
importlib.reload(grp)
importlib.reload(cp)

FILE_NAME_LAYOUT = "results/ssm_nodePos_kamadaKawai_2017-12.gpickle"
# FILE_NAME_LAYOUT = "results/ssm_nodePos_spring_2017-12.gpickle"
NODE_SIZE = 10
NODE_ALPHA = 0.75
LABEL_SIZE = 7
FIG_SiZE = 4

sns.set_style("darkgrid")
sns.set_context("notebook")

# Load position of nodes
pos = nx.read_gpickle(FILE_NAME_LAYOUT)

# Group by actors by party for node colouring
groupedActors, colourMap, legend = grp.byAttr(G, 'Party')
# groupedActors, colourMap, legend = grp.byAttr(G, 'Gender')
# groupedActors, colourMap, legend = grp.byAttr(G, 'Metro')

# ----- Draw nodes
# TODO: Vary node size according ot its centrality
# TODO: Write function to colour nodes by single colour (default gray)
# TODO: Write function to colour nodes base on attribute
fig = plt.figure(figsize=(FIG_SiZE*3, FIG_SiZE*2), dpi=300, tight_layout=True)
plt.title(f"Same-sex Marriage Bill {FILE_NAME[-11:-4]}")

for (g, lab) in zip(list(groupedActors.keys()), legend):
    nx.draw_networkx_nodes(G, pos, nodelist=groupedActors[g], node_size=NODE_SIZE*100, node_color=colourMap[g], alpha=NODE_ALPHA, edgecolors='black', linewidths=1, label=lab)
plt.legend()

# ----- Draw node labels
# Construct dict to reformat names from 'first last' to 'first\nlast'
nodes = G.nodes
nodeLabels = []
for n in nodes:
    nodeLabels.append(re.sub(' ', '\n', n))
nodeLabels = dict(zip(nodes, nodeLabels))

nx.draw_networkx_labels(G, pos, labels=nodeLabels, font_size=LABEL_SIZE, font_color='black')
# Draw edges
# TODO: Write function to colour edges with thickness unweighted and a single colour (default gray)
# TODO: Write function to colour edges with thickness weighted and a single colour (default gray)

nx.draw_networkx_edges(G, pos, width=0.25, edge_color=cp.cbDark2['gray'])


# ----- Save figure
FORMAT = '.png'
graphFileName = re.sub('.csv', FORMAT, FILE_NAME)
graphFileName = re.sub('results', 'graph', graphFileName)

# fig.savefig(f"results/{graphFileName}", dpi=300)
plt.show()


#%% Draw Clique Graph
# TODO: Draw clique graphs manually
NODE_SIZE = 8
NODE_ALPHA = 0.75
FIG_SiZE = 4

sns.set_style("darkgrid")
sns.set_context("notebook")

CG = nx.Graph()
CG.clear()



cliques

#%% Save figure
FORMAT = '.png'
graphFileName = re.sub('.csv', FORMAT, FILE_NAME)
graphFileName = re.sub('results', 'cliques', graphFileName)
fig.savefig(f"results/{graphFileName}", dpi=300)
plt.show()


#%%
# TODO: Plot histogram of all centrality first to see the distribution
# TODO: Plot centrality of politicians against time
