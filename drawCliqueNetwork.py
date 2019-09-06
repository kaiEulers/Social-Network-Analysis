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
import json
import groupActor as grp
import colourPals as cp


#%% Draw Clique Network
# TODO: Draw clique graphs manually!!!
importlib.reload(grp)
importlib.reload(cp)

# ----- Load graph, centrality, and clique data
DATE = '2017-12'
FILE_NAME = f"ssm_{DATE}_results_NMF_senti.csv"
FILE_NAME_GRAPH = FILE_NAME.replace('NMF_senti.csv', 'graph.png')

results = pd.read_csv(f"results/{FILE_NAME}")
G = nx.read_gpickle(f"results/{FILE_NAME_GRAPH.replace('.png', '.gpickle')}")
# TODO: Load normal graph with weighted edges
cent = pd.read_csv(f"results/{FILE_NAME.replace('NMF_senti', 'centrality')}")
with open(f"results/{FILE_NAME.replace('NMF_senti.csv', 'cliques.json')}", "r") as file:
    cliques = json.load(file)

FILE_NAME_LAYOUT = f"results/ssm_{DATE}_nodePos_kamadaKawai.gpickle"
# FILE_NAME_LAYOUT = "results/ssm_{DATE}_nodePos_spring.gpickle"
NODE_SIZE = 10
NODE_ALPHA = 0.75
LABEL_SIZE = 7
FIG_SiZE = 4

sns.set_style("darkgrid")
sns.set_context("notebook")




#%% Save figure
fig.savefig(f"results/{FILE_NAME_GRAPH.replace('graph', 'cliqueGraphs')}", dpi=300)
plt.show()

# TODO: Plot histogram of all centrality first to see the distribution
# TODO: Plot centrality of politicians against time