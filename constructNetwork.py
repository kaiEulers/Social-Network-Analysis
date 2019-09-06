"""
@author: kaisoon
"""
# ----- Imports
import os
import numpy as np
import pandas as pd
import networkx as nx
import json


#%% ----- Compute sentiment threshold
# If the difference in sentiment is higher than the threshold, both speeches do not agree with each other.
# TODO: Determine sentiDiff_thres statistically & empirically. Try significant levels of 0.01, 0.05, 0.1
# TODO: Find all sentiment difference between all speeches plot its distribution
sentiDiff_thres = 2


#%% ---------- Construct Network Graph
DATE = '2017-12'
FILE_NAME = f"ssm_{DATE}_results_NMF_senti.csv"
results = pd.read_csv(f"results/{FILE_NAME}")

G = nx.MultiGraph()
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
            Gender=row['Gender'],
            Party=row['Party'],
            Metro=row['Metro'],
            Data=data
        )
        # Print progress...
        if i%50 == 0:
            print(f"{i:{5}} of {len(results):{5}}\t{dict(row['Speech_id Date Person Party'.split()])}")
print('All nodes succesfully added!')


# ----- Add edges
print('\nAdding edges...')
for i in range(len(results)):
    row_i = results.iloc[i]
    # Extract name, topic and sentiment of person1
    p1 = row_i['Person']
    t1 = row_i['Topic_nmf']
    s1 = row_i['Senti_comp']

    for j in range(i, len(results)):
        row_j = results.iloc[j]
        # Extract name, topic and sentiment of person2
        p2 = row_j['Person']
        t2 = row_j['Topic_nmf']
        s2 = row_j['Senti_comp']

        # person1 and person2 cannot be the same person
        if p1 != p2:
            # Both actors must spoke of the same topic have have the same polarity of sentiment on the topic. Note: if s1*s2 > 0 then both senitmetn are of the same polarity.
            if (t1 == t2) and (s1*s2 > 0):
                # Compute sentiment difference
                sentiDiff = abs(s1 - s2)
                # Both actors must have sentiment towards the topic that is less than the threshold
                if sentiDiff < sentiDiff_thres:
                    # Connect both actors if all conditions above are true
                    G.add_edge(
                        p1, p2,
                        Topic=row_i['Topic_nmf'],
                        SentiDiff=sentiDiff,
                        Data=pd.DataFrame([row_i, row_j])
                    )
    # Print progress...
    if i%10 == 0:
        print(f"{i:{5}}, {j:{5}} of {len(results):{5}}\t{p1:{20}}{p2:{20}}\tt1:{t1}\tt2:{t2}")
print('All edges succesfully added!')


# ----- Compute centrality and add as node attribute
cent = pd.DataFrame(G.degree(), columns='Person Edge_Count'.split())
cent.set_index('Person', inplace=True)
cent.sort_values('Edge_Count', ascending=False, inplace=True)
cent_max = cent.max().iloc[0]
cent['Degree'] = cent['Edge_Count']/cent_max

# Add centrality information to node attribute
nx.set_node_attributes(G, cent['Degree'], 'Centrality')
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
with open(f"results/{FILE_NAME.replace('NMF_senti.csv', 'cliques.json')}", "w") as file:
    json.dump(networkCliques, file)

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
nx.set_node_attributes(G, cliqueNum, 'Cliques')


# ----- Save graph
nx.write_gpickle(G, f"results/{FILE_NAME.replace('NMF_senti.csv', 'graph.gpickle')}")


# TODO: Construct normal graph with weighted edges using multigraph

# os.system('say Complete')

#%% Examine centrality data
import matplotlib.pyplot as plt
import seaborn as sns



sns.distplot(pd.Series(cent), bins=5, kde=False, norm_hist=True)
plt.show()