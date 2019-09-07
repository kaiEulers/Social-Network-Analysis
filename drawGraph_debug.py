"""
@author: kaisoon
"""
# ----- Imports
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import importlib
import group
import colourPals as cp
import time


#%% Get node position in kamada kawai layout
startTime = time.time()
print('Starting layout generation...')
DATE = '2017-12'
FILE_NAME = f"ssm_{DATE}_results_NMF_senti.csv"
G = nx.read_gpickle(f"results/{FILE_NAME.replace('NMF_senti.csv', 'weightedGraph.gpickle')}")

# TODO: Need to play around with graph drawing algorithms
# TODO: Try different layout for clique graph
pos = nx.kamada_kawai_layout(G, weight=None)
# Save position coordinates
nx.write_gpickle(pos, f"results/ssm_{DATE}_nodePos_kamadaKawai.gpickle")


# Use kamadaKawai layout positions as initial position for spring layout
# Increase k to spread nodes further apart
pos = nx.spring_layout(G, pos=pos, iterations=100, k=10)
# Save position coordinates
nx.write_gpickle(pos, f"results/ssm_{DATE}_nodePos_spring.gpickle")

print(f"\nLayout generation complete! Generation took {time.time()-startTime}s!")


#%% ----- Draw Network
startTime = time.time()
print('Drawing graph...')
importlib.reload(group)
importlib.reload(cp)

# TODO: How should I determine the centrality threshold?
CENT_THRES = 0.75

# ----- Set up graph layout
FILE_NAME_LAYOUT = f"results/ssm_{DATE}_nodePos_kamadaKawai.gpickle"
# FILE_NAME_LAYOUT = f"results/ssm_{DATE}_nodePos_spring.gpickle"
# Available attributes: party, gender, metro
ATTRIBUTE = 'party'
NODE_SIZE = 11
NODE_ALPHA = 0.85
NODE_LABEL_SIZE = 7
EDGE_WIDTH = 0.25
LEGEND_FONT_SIZE = 9
FIG_SiZE = 4

sns.set_style("darkgrid")
sns.set_context("notebook")

# ----- Load graph
DATE = '2017-12'
FILE_NAME = f"ssm_{DATE}_results_NMF_senti.csv"
FILE_NAME_GRAPH = FILE_NAME.replace('NMF_senti.csv', 'weightedGraph.png')

G = nx.read_gpickle(f"results/{FILE_NAME.replace('NMF_senti.csv', 'weightedGraph.gpickle')}")
# Load position of nodes
pos = nx.read_gpickle(FILE_NAME_LAYOUT)


# ----- Draw nodes
print("\nDrawing nodes...")
fig = plt.figure(figsize=(FIG_SiZE*3, FIG_SiZE*2), dpi=300, tight_layout=True)
plt.title(f"Same-sex Marriage Bill {DATE}")

# Group by actors by party for node colouring
groupedActors, cm_nodes, leg_nodes = group.byNodeAttr(G, ATTRIBUTE)
for grp in groupedActors.keys():
    nx.draw_networkx_nodes(G, pos, nodelist=groupedActors[grp], node_size=NODE_SIZE*100, node_color=cm_nodes[grp], alpha=NODE_ALPHA, edgecolors='black', linewidths=0.5, label=leg_nodes[grp])
print("Node drawing complete!")


# ----- Draw node labels
print("\nDrawing node labels...")
# Group nodeLabels by actors with high centrality
groupedLabels, cm_labels, sm_labels, fwm_labels = group.byCent4NodeLabel(G, CENT_THRES)
for grp in groupedLabels.keys():
    nx.draw_networkx_labels(G, pos, labels=groupedLabels[grp], font_size=NODE_LABEL_SIZE*sm_labels[grp], font_color=cm_labels[grp], font_weight=fwm_labels[grp])
print("Node label drawing complete!")


# ----- Draw edges
print("\nDrawing edges...")
groupedEdges, cm_edges, sm_egdes, leg_edges = group.byEdgeWeight(G)
for grp in groupedEdges.keys():
    nx.draw_networkx_edges(G, pos, edgelist=groupedEdges[grp], width=EDGE_WIDTH*sm_egdes[grp], edge_color=cm_edges[grp], label=leg_edges[grp])
print("Edge drawing complete!")


# ----- Draw legend
plt.legend(markerscale=LEGEND_FONT_SIZE*0.05, fontsize=LEGEND_FONT_SIZE)


# ----- Save figure
fig.savefig(f"results/{FILE_NAME_GRAPH}", dpi=300)
plt.show()
print(f"\nGraph drawing complete! Drawing took {time.time()-startTime}s!")

