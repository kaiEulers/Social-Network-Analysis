# ----- Imports
import random as rand
import numpy as np
import networkx as nx

DATE = "2017-12"
G = nx.read_gpickle(f"results/ssm_results_graph_{DATE}.gpickle")

FILE_NAME = "results/realityChecks/"

# Extract data from graph
actorList = np.sort(list(G.node))
nodeData = dict(G.nodes.data())
edgeData = list(G.edges.data())


#%% Reality check for node data
actor = rand.choice(actorList)
print(f"Actor: {actor}")
actorData = nodeData[actor]['Data'].T
print(f"{actorData.shape[1]} speeches")


# Save actorData
for n,k in zip(actorData.columns, range(actorData.shape[1])):
    actorData[n].to_csv(f"{FILE_NAME}node/{actor.replace(' ', '')}-{k}.csv", header=False)


#%% Reality check of edge data
i = rand.randint(0, len(edgeData))
print(f"Edge#: {i}")

edge = edgeData[i]
actor1 = edge[0]
actor2 = edge[1]
print(edge[0], '-', edge[1])

edgeAttr = edge[2]
relationData = edgeAttr['Data'].T

# Save relationData
for n,k in zip(relationData.columns, range(relationData.shape[1])):
    relationData[n].to_csv(f"{FILE_NAME}edge/{actor1.replace(' ', '')}-{actor2.replace(' ', '')}-{k}.csv", header=False)

