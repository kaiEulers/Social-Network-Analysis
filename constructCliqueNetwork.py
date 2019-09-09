"""
@author: kaisoon
"""
# ----- Imports
import pandas as pd
import networkx as nx
import pickle
import time


# =====================================================================================
# Load graph, centrality, and clique data
startTime = time.time()
DATE = '2017-12'
PATH = f"results/{DATE}/"

results = pd.read_csv(f"{PATH}ssm_{DATE}_results_NMF_senti.csv")
G = nx.read_gpickle(f"{PATH}ssm_{DATE}_weightedGraph.gpickle")
with open(f"{PATH}ssm_{DATE}_cliques.pickle", "rb") as file:
    cliqueDict = pickle.load(file)

# ----- Construct clique graphs
# Get centrality and cliqueNum data
centDict = nx.get_node_attributes(G, 'centrality')
cliqueNumDict = nx.get_node_attributes(G, 'cliques')

CGs = {}
for k in cliqueDict.keys():
    CG = nx.Graph()
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
    print(f"{len(CGs)} were cliques found")


# Save clique graph
nx.write_gpickle(CGs, f"{PATH}ssm_{DATE}_cliqueGraphs.gpickle")
print(f"\nClique graph construction complete! Construction took {round(time.time()-startTime, 2)}s")

