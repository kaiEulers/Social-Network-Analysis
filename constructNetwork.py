"""
@author: kaisoon
"""
# ----- Imports
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import time


#%% ----- Compute sentiment threshold
# If the difference in sentiment is higher than the threshold, both speeches do not agree with each other.
# TODO: Determine sentiDiff_thres statistically & empirically. Try significant levels of 0.01, 0.05, 0.1
# TODO: Find all sentiment difference between all speeches plot its distribution
sentiDiff_thres = 2


#%% ----- Construct Weighted Graph
startTime = time.time()
DATE = '2017-12'
FILE_NAME = f"ssm_{DATE}_results_NMF_senti.csv"
results = pd.read_csv(f"results/{FILE_NAME}")

G = nx.Graph()
G.clear()

# ----- Add politicians as nodes in the graph
print('\nAdding nodes...')
for i in results.index:
    row = results.loc[i]
    person = row['Person']
    # Only add actor if the actor hasn't already been added
    if not G.has_node(person):
        # Construct dataFrame for data attribute of node
        # Extract all data from the actor
        data = results[results['Person'] == person]
        data.index = range(len(data))

        # Add node with its corresponding attributes
        G.add_node(
            person,
            gender=row['Gender'],
            party=row['Party'],
            metro=row['Metro'],
            data=data
        )
        # Print progress...
        if i % 50 == 0:
            print(f"{i:{5}} of {len(results):{5}}\t{dict(row['Speech_id Date Person Party'.split()])}")
print('All nodes succesfully added!')

print('\nProcessing edges...')
for i in range(len(results)):
    row_i = results.iloc[i]
    # Extract name, topic and sentiment of person1
    p1 = row_i['Person']
    t1 = row_i['Topic_nmf']
    s1 = row_i['Senti_comp']

    for j in range(i+1, len(results)):
        row_j = results.iloc[j]
        # Extract name, topic and sentiment of person2
        p2 = row_j['Person']
        t2 = row_j['Topic_nmf']
        s2 = row_j['Senti_comp']

        # Print progress...
        if (i % 10 == 0) and (j % 50 == 0):
            print(
                f"{i:{5}},{j:{5}} of {len(results):{5}}\t{p1:{20}}{p2:{20}}\tt1: {int(t1)}\tt2: {int(t2)}")

        # Both actors must spoke of the same topic and cannot be the same person
        if (t1 == t2) and (p1 != p2):
            # Compute sentiment difference
            sentiDiff = abs(s1 - s2)
            # Both actors must have sentiment towards the topic that is less than the threshold and of the same polarity
            if (sentiDiff < sentiDiff_thres) and (s1*s2 > 0):
                # If there is no edge between both actors, construct an edge. Otherwise, update attribtes of the existing edge.
                if not G.has_edge(p1, p2):
                    agreedSpeeches = {
                        'topic': t1,
                        'sentiDiff': sentiDiff,
                        'data': pd.DataFrame([row_i, row_j])
                    }
                    G.add_edge(p1, p2, weight=1, agreedSpeeches=[agreedSpeeches])
                else:
                    # Extract data from already existing edge
                    edgeData = G.get_edge_data(p1, p2)

                    # Update weight data
                    weight_old = edgeData['weight']
                    weight_new = weight_old + 1
                    # Update agreedSpeeches
                    agreedSpeeches = edgeData['agreedSpeeches']
                    agreedSpeeches.append(pd.DataFrame([row_i, row_j]))

                    # Update information of the edge
                    G.add_edge(p1, p2, weight=weight_new, agreedSpeeches=agreedSpeeches)
print('All edges succesfully added!')


# ----- Compute centrality and add as node attribute
cent = {}
# G.adjacency() returns iterator over nodes and a dict containing names of all nodes adjacent to the node. Value of the dict contains edge attribute between these two nodes.
for node,adjDict in G.adjacency():
    # Sum up all weight attributes of each edge connected to the node
    edgeWeightSum = 0
    for p in adjDict.keys():
        edgeWeightSum += adjDict[p]['weight']
    cent[node] = edgeWeightSum

# Place data in dataFrame and sort according to edgeWeightSum
cent = pd.DataFrame.from_dict(cent, orient='index', columns='EdgeWeightSum'.split())
cent.sort_values('EdgeWeightSum', ascending=False, inplace=True)
# Compute degree of centrality with respect to max edgeWeightSum
cent_max = cent['EdgeWeightSum'].max()
cent['Degree'] = cent['EdgeWeightSum']/cent_max

# Add centrality information to node attribute
nx.set_node_attributes(G, cent['Degree'], 'centrality')

# Save centrality results
cent.to_csv(f"results/{FILE_NAME.replace('NMF_senti', 'centrality')}")


# ----- Compute cliques and add clique group number as node attribute
# Construct a dictionary containing cliques within the network labeled by a clique#
networkCliques = {}
num = 0
for c in nx.find_cliques(G):
    networkCliques[num] = c
    num += 1
# Save clique results
# with open(f"results/{FILE_NAME.replace('NMF_senti.csv', 'cliques.json')}", "w") as file:
#     json.dump(networkCliques, file)
with open(f"results/{FILE_NAME.replace('NMF_senti.csv', 'cliques.pickle')}", "wb") as file:
    pickle.dump(networkCliques, file)


# For every actor in the network, search all networkCliques to find if the actor is in it
# Return a dict of actors and the corresponding clique# that the actor is in
cliqueNum = {}
actors = np.sort(list(G.node))
for p in actors:
    inClique = []
    for i in range(len(networkCliques)):
        if p in networkCliques[i]:
            inClique.append(i)
    cliqueNum[p] = inClique

# Add clique information to node attribute
nx.set_node_attributes(G, cliqueNum, 'cliques')


# ----- Save graph
nx.write_gpickle(G, f"results/{FILE_NAME.replace('NMF_senti.csv', 'weightedGraph.gpickle')}")
print(f"\nGraph construction complete! Construction took {round(time.time()-startTime, 2)}s")


#%% ---------- Construct Multi-Graph
startTime = time.time()
DATE = '2017-12'
FILE_NAME = f"ssm_{DATE}_results_NMF_senti.csv"
results = pd.read_csv(f"results/{FILE_NAME}")

MG = nx.MultiGraph()
MG.clear()

# ----- Add politicians as nodes in the graph
print('\nAdding nodes...')
for i in results.index:
    row = results.loc[i]
    person = row['Person']
    # Only add actor if the actor hasn't already been added
    if not MG.has_node(person):
        # Construct dataFrame for data attribute of node
        # Extract all data from the actor
        data = results[results['Person'] == person]
        data.index = range(len(data))

        # Add node with its corresponding attributes
        MG.add_node(
            person,
            gender=row['Gender'],
            party=row['Party'],
            metro=row['Metro'],
            data=data
        )
        # Print progress...
        if i % 50 == 0:
            print(f"{i:{5}} of {len(results):{5}}\t{dict(row['Speech_id Date Person Party'.split()])}")
print('All nodes succesfully added!')


# ----- Add edges
print('\nProcessing edges...')
for i in range(len(results)):
    row_i = results.iloc[i]
    # Extract name, topic and sentiment of person1
    p1 = row_i['Person']
    t1 = row_i['Topic_nmf']
    s1 = row_i['Senti_comp']

    for j in range(i+1, len(results)):
        row_j = results.iloc[j]
        # Extract name, topic and sentiment of person2
        p2 = row_j['Person']
        t2 = row_j['Topic_nmf']
        s2 = row_j['Senti_comp']

        # Print progress...
        if (i % 10 == 0) and (j % 50 == 0):
            print(
                f"{i:{5}},{j:{5}} of {len(results):{5}}\t{p1:{20}}{p2:{20}}\tt1: {int(t1)}\tt2: {int(t2)}")

        # person1 and person2 cannot be the same person
        if p1 != p2:
            # Both actors must spoke of the same topic have have the same polarity of sentiment on the topic. Note: if s1*s2 > 0 then both senitmetn are of the same polarity.
            if (t1 == t2) and (s1*s2 > 0):
                # Compute sentiment difference
                sentiDiff = abs(s1 - s2)
                # Both actors must have sentiment towards the topic that is less than the threshold
                if sentiDiff < sentiDiff_thres:
                    # Connect both actors if all conditions above are true
                    MG.add_edge(
                        p1, p2,
                        topic=row_i['Topic_nmf'],
                        sentiDiff=sentiDiff,
                        data=pd.DataFrame([row_i, row_j])
                    )
print('All edges succesfully added!')


# ----- Save multi-graph
nx.write_gpickle(MG, f"results/{FILE_NAME.replace('NMF_senti.csv', 'multigraph.gpickle')}")
print(f"\nGraph construction complete! Construction took {round(time.time()-startTime, 2)}s")


#%% Plot distribution of centrality
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
sns.set_context("notebook")

cent = pd.read_csv(f"results/{FILE_NAME.replace('NMF_senti', 'centrality')}")
cent.shape

fig = plt.figure(dpi=300, tight_layout=True)
ax = sns.distplot(pd.Series(cent['EdgeWeightSum']), kde=False)
ax.set_title(f"Same-sex Marriage Bill {DATE}\nCentrality")
ax.set_ylabel("Frequency")
fig.savefig(f"results/{FILE_NAME.replace('NMF_senti.csv', 'centrality.png')}", dpi=300)
fig.show()


#%% Draw graph
import kaiGraph as kg
import importlib
importlib.reload(kg)

fig = kg.drawGraph(G)
fig.show()

