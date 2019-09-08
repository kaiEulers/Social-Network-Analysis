"""
@author: kaisoon
"""
# ----- Imports
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import importlib
import kaiGraph as kg
import kaiGraph1 as kg1

sns.set_style("darkgrid")
sns.set_context("notebook")

DATE = '2017-12'
FIG_SiZE = 4
SUBGRAPH_LAYOUT = 221
FILE_NAME = f"ssm_{DATE}_results_weightedGraph.gpickle"
G = nx.read_gpickle(f"results/{FILE_NAME}")


#%% Normal Graph Drawing Test
importlib.reload(kg)

GROUPBY = 'party'
LAYOUT = 'kamada'

fig = plt.figure(figsize=(FIG_SiZE * 3, FIG_SiZE * 2), dpi=300, tight_layout=True)
kg.drawGraph(G, groupBy=GROUPBY, layout=LAYOUT,
             title=f"Same-sex Marriage Bill {DATE}")
plt.show()



#%% Graph Drawing Test with attempted grouping of nodes by centrality
importlib.reload(kg)

GROUPBY = 'party'
LAYOUT = 'kamada'

fig = plt.figure(figsize=(FIG_SiZE * 3, FIG_SiZE * 2), dpi=300, tight_layout=True)
kg1.drawGraph(G, groupBy=GROUPBY, layout=LAYOUT,
              title=f"Same-sex Marriage Bill {DATE}")
plt.show()

fig.savefig(f"results/{FILE_NAME.replace('weightedGraph.gpickle', 'graph.png')}", dpi=300)
fig.show()


#%% Normal Clique Graph Drawing Test
importlib.reload(kg)

GROUPBY = 'party'
LAYOUT = 'spring'

fig = plt.figure(figsize=(FIG_SiZE * 3, FIG_SiZE * 2), dpi=300, tight_layout=True)
# Draw cliqueGraphs in subplots
for k,gph in CGs.items():
    plt.subplot(SUBGRAPH_LAYOUT + k)
    kg.drawGraph(gph, groupBy=GROUPBY, layout=LAYOUT,
                 title=f"{DATE} Clique{k}",
                 legend=False,
                 node_size=5,
                 font_size=6,
                 )
plt.show()


#%% Clique Graph Drawing Test with attempted grouping of nodes by centrality
importlib.reload(kg1)

GROUPBY = 'party'
LAYOUT = 'spring'

fig = plt.figure(figsize=(FIG_SiZE * 3, FIG_SiZE * 2), dpi=300, tight_layout=True)
# Draw cliqueGraphs in subplots
for k,gph in CGs.items():
    plt.subplot(SUBGRAPH_LAYOUT + k)
    kg1.drawGraph(gph, groupBy=GROUPBY, layout=LAYOUT,
                 title=f"{DATE} Clique{k}",
                 legend=False,
                 node_size=5,
                 font_size=6,
                 node_size_highCent= 10,
                 )
plt.show()

fig.savefig(f"results/{FILE_NAME.replace('weightedGraph.gpickle', 'cliqueGraphs.png')}", dpi=300)
fig.show()


