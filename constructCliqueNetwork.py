"""
@author: kaisoon
"""
# ----- Imports
import numpy as np
import pandas as pd
import networkx as nx
import json


#%% ----- Load graph, centrality, and clique data
DATE = '2017-12'
FILE_NAME = f"ssm_{DATE}_results_NMF_senti.csv"
FILE_NAME_GRAPH = FILE_NAME.replace('NMF_senti.csv', 'graph.gpickle')

results = pd.read_csv(f"results/{FILE_NAME}")
G = nx.read_gpickle(f"results/{FILE_NAME_GRAPH}")
# TODO: Load normal graph with weighted edges

cent = pd.read_csv(f"results/{FILE_NAME.replace('NMF_senti', 'centrality')}")
with open(f"results/{FILE_NAME.replace('NMF_senti.csv', 'cliques.json')}", "r") as file:
    cliques = json.load(file)


# ----- Construct clique network
CG = nx.Graph()
CG.clear()

cent = dict(G.nodes.data('Centrality'))
cliqueNum = dict(G.nodes.data('Cliques'))

for k in cliques.keys():
    actors = cliques[k]
    # ----- Add nodes
    print(f"\nAdding nodes for cliqueGraph{k}...")
    for person in actors:
        # Extract all data on this person
        data = results[results['Person'] == person]
        # Extract one row from the data
        row = data.iloc[0]
        CG.add_node(
            person,
            Gender=row['Gender'],
            Party=row['Party'],
            Metro=row['Metro'],
            Data=data,
            Centrality=cent[person],
            Cliques=cliqueNum[person]
        )
    print(f"All nodes successfully added for cliqueGraph{k}")

    # ----- Add edges
    print(f"\nAdding edges for cliqueGraph{k}...")
    for i in range(len(actors)):
        p1 = actors[i]
        for j in range(i, len(actors)):
            p2 = actors[j]
            CG.add_edge(
                p1, p2
            #     TODO: Add edge data with fully constructed graph
            )
    print(f"All edges successfully added for cliqueGraph{k}")

    nx.write_gpickle(CG, f"results/{FILE_NAME_GRAPH.replace('graph', 'cliqueGraph' + k)}")
