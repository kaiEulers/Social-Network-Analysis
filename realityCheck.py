# ----- Imports
import pandas as pd
import random as rand
import numpy as np
import networkx as nx

DATE = "2017"
nTOPICS = 5
data = pd.read_csv(f"self/ssm_rel.csv")
results = pd.read_csv(f"results/ssm_results_nmf_senti.csv")
topics = pd.read_csv(f"results/ssm_{nTOPICS}topics_nmf.csv", index_col=0)
G = nx.read_gpickle(f"results/{DATE}/ssm_weightedGraph_{DATE}.gpickle")

PATH = "results/realityChecks/"

# Extract all speeches
data['Speech'].to_csv(f"{PATH}allSpeeches.csv")

# Extract self from graph
actorList = np.sort(list(G.node))
nodeData = dict(G.nodes.data())
edgeData = list(G.edges.data())


#%% Reality check for node self
actor = rand.choice(actorList)
print(f"Actor: {actor}")
actorData = nodeData[actor]['Data'].T
print(f"{actorData.shape[1]} speeches")

# Save topicProfiles
for n,k in zip(actorData.columns, range(actorData.shape[1])):
    actorData[n].to_csv(f"{PATH}node/{actor.replace(' ', '')}-{k}.csv", header=False)


#%% Reality check of edge self
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
    relationData[n].to_csv(f"{PATH}edge/{actor1.replace(' ', '')}-{actor2.replace(' ', '')}-{k}.csv", header=False)


#%% Extract speech with highest coeffMax
speech_id = topics.loc['speechId']
highestCoeffMax = results[results['Speech_id'].isin(speech_id)]
highestCoeffMax = highestCoeffMax.sort_values(by='Topic')

for i, row in highestCoeffMax.iterrows():
    row_T = row.transpose()
    row_T.to_csv(f"{PATH}highestMaxCoeff/highestCoeffMax_topic{row['Topic']}.csv", header=True)

#%% Extract particular speech of a topic
TOPIC = 8
k = 0

speech = results[results['Topic'] == TOPIC].iloc[k]
speech = speech.transpose()
speech.to_csv(f"{PATH}speech.csv", header=True)

#%% Extract particular speech
id = 41941
i = 141

# speech = self[self['Speech_id'] == id]
speech = data.loc[i]
speech = speech.transpose()
speech.to_csv(f"{PATH}speech.csv", header=True)


#%% Check if word is in any of the speeches
word = 'plebiscite'
ids = []
for i, row in results.iterrows():
    if word in row['Speech'].lower():
        ids.append(i)
print(ids)
