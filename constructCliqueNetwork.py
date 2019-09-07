"""
@author: kaisoon
"""
# ----- Imports
import numpy as np
import pandas as pd
import networkx as nx
import json
import time
import drawGraph
import importlib


#%% ----- Load graph, centrality, and clique data
startTime = time.time()
DATE = '2017-12'
FILE_NAME = f"ssm_{DATE}_results_NMF_senti.csv"
FILE_NAME_GRAPH = FILE_NAME.replace('NMF_senti.csv', 'weightedGraph.gpickle')

results = pd.read_csv(f"results/{FILE_NAME}")
G = nx.read_gpickle(f"results/{FILE_NAME_GRAPH}")
with open(f"results/{FILE_NAME.replace('NMF_senti.csv', 'cliques.json')}", "r") as file:
    cliqueDict = json.load(file)

# ----- Construct clique graphs
# Get centrality and cliqueNum data
centDict = nx.get_node_attributes(G, 'centrality')
cliqueNumDict = nx.get_node_attributes(G, 'cliques')

CGs = {}
CG = nx.Graph()
for k in cliqueDict.keys():
    CG.clear()
    clique = cliqueDict[k]

    # ----- Add nodes
    # TODO: Try adding nodes directly from loaded graph G
    print(f"\nAdding nodes for cliqueGraph{k}...")
    for person in clique:
        # Extract all data of this person
        data = results[results['Person'] == person]
        # Extract one row from the data
        row = data.iloc[0]
        CG.add_node(
            person,
            gender=row['Gender'],
            party=row['Party'],
            metro=row['Metro'],
            data=data,
            centrality=centDict[person],
            cliques=cliqueNumDict[person]
        )
    print(f"All nodes successfully added for cliqueGraph{k}")

    # ----- Add edges
    print(f"Adding edges for cliqueGraph{k}...")
    for i in range(len(clique)):
        p1 = clique[i]
        for j in range(i+1, len(clique)):
            p2 = clique[j]
            # Check that p1 and p2 are not the same people
            if p1 != p2:
                # Get edge data
                edgeData = G.get_edge_data(p1, p2)
                CG.add_edge(p1, p2, weight=edgeData['weight'], agreedSpeeches=edgeData['agreedSpeeches'])

    CGs[k] = CG
    print(f"All edges successfully added for cliqueGraph{k}")


# Save clique graph
nx.write_gpickle(CGs, f"results/{FILE_NAME_GRAPH.replace('graph', 'cliqueGraph' + k)}")
print(f"\nClique graph construction complete! Construction took {time.time()-startTime}s!")


#%%
# TODO: Draw clique graphs
importlib.reload(drawGraph)

for CG in CGs.keys():
    fig = drawGraph.draw(CG)
    fig.show()

