"""
@author: kaisoon
"""
# ----- Imports
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import time


# =====================================================================================
# Compute sentiment difference threshold
DATE = '2017-12'
PATH = f"results/{DATE}/"
# TODO: How do I determine the sentiment diff threshold? WE WANT SENTI DIFF THAT ARE LOW!
SENTIDIFF_PERC_THRES = 0.2
sentiDiff = pd.read_csv(f"{PATH}distributions/ssm_{DATE}_sentiDiff.csv")

sentiDiff_sig = sentiDiff[sentiDiff['Percentile'] < 0.2]
SENTIDIFF_THRES = sentiDiff_sig['Senti_Diff'].max()


# =====================================================================================
# Construct Weighted Graph
startTime = time.time()
results = pd.read_csv(f"{PATH}ssm_{DATE}_results_NMF_senti.csv")

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

        # Both actors cannot be the same person
        # Both actors must spoke of the same topic
        # Both sentiment of the same topic must be of the same polarity
        if (p1 != p2) and (t1 == t2) and (s1*s2 > 0):
            # Compute sentiment difference
            sentiDiff = abs(s1 - s2)
            # Both sentiment towards the topic must be less than the threshold
            if (sentiDiff < SENTIDIFF_THRES):
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


# =====================================================================================
# Compute centrality and add as node attribute
cent = {}
# G.adjacency() returns iterator over nodes and a dict containing names of all nodes adjacent to the node. Value of the dict contains edge attribute between these two nodes.
for node,adjDict in G.adjacency():
    # Sum up all weight attributes of each edge connected to the node
    edgeWeightSum = 0
    for p in adjDict.keys():
        edgeWeightSum += adjDict[p]['weight']
    cent[node] = edgeWeightSum

# Place data in dataFrame and sort according to edgeWeightSum
cent = pd.DataFrame.from_dict(cent, orient='index', columns='Degree'.split())
cent.sort_values('Degree', ascending=False, inplace=True)
# Compute degree of centrality with respect to max edgeWeightSum
cent_max = cent['Degree'].max()
cent['Percentile'] = cent['Degree']/cent_max

# Add centrality information to node attribute
nx.set_node_attributes(G, cent['Percentile'], 'centrality')

# Save centrality results
cent.to_csv(f"{PATH}ssm_{DATE}_centrality.csv")


# =====================================================================================
# Compute cliques and add clique group number as node attribute
# Construct a dictionary containing cliques within the network labeled by a clique#
networkCliques = {}
num = 0
for c in nx.find_cliques(G):
    networkCliques[num] = c
    num += 1
# Save clique results
# with open(f"results/{FILE_NAME.replace('NMF_senti.csv', 'cliques.json')}", "w") as file:
#     json.dump(networkCliques, file)
with open(f"{PATH}ssm_{DATE}_cliques.pickle", "wb") as file:
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
nx.write_gpickle(G, f"{PATH}ssm_{DATE}_weightedGraph.gpickle")

print(f"\nGraph construction complete! Construction took {round(time.time()-startTime, 2)}s")
# Print percentage of edges removed by threshold
print(f"Percentage of edges removed by sentiDiff threshold: {round((len(sentiDiff)-len(sentiDiff_sig))/len(sentiDiff), 4)*100}%")