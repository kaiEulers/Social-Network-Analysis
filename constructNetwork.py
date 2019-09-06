"""
@author: kaisoon
"""
#%% ----- Imports
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import re

FILE_NAME = "ssm_results_NMF_senti_2017-12.csv"
results = pd.read_csv(f"results/{FILE_NAME}")


#%% ---------- Construct Network Graph
# If the difference in sentiment is higher than the threshold, both speeches do not agree with each other.
# TODO: Determine sentiDiff_thres statistically & empirically. Try significant levels of 0.01, 0.05, 0.1
# TODO: Find all sentiment difference between all speeches plot its distribution
sentiDiff_thres = 2

G = nx.MultiGraph()
G.clear()

# ----- Add politicians as nodes in the graph
print('\nAdding nodes...')
for i in results.index:
    # Print progress...
    if i % 10 == 0:
        print(f"{i} of {len(results)}")

    row = results.loc[i]
    person = row['Person']
    # Only add actor if the actor hasn't already been added
    if not G.has_node(person):
        # Construct dataFrame for speechData attribute of node
        # Extract all results from the actor
        speechData = results[results['Person'] == person]
        speechData.index = range(len(speechData))
        # Drop unnecessary columns
        speechData.drop('Person Gender Party Elec Metro'.split(), axis=1, inplace=True)

        # Add node with its corresponding attributes
        G.add_node(
            person,
            Gender=row['Gender'],
            Party=row['Party'],
            Elec=row['Elec'],
            Metro=row['Metro'],
            SpeechData=speechData
        )
print('All nodes succesfully added!')


# ----- Add edges
print('\nAdding edges...')
for i in results.index:
    # Print progress...
    if i % 10 == 0:
        print(f"{i} of {len(results)}")

    row_i = results.loc[i]
    # Extract name, topic and sentiment of person1
    p1 = row_i['Person']
    t1 = row_i['Topic_nmf']
    s1 = row_i['Senti_comp']

    for j in results.index:
        row_j = results.loc[j]
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
                        SpeechData=pd.DataFrame([row_i, row_j])
                    )
print('All edges succesfully added!')

# ----- Compute centrality and add as node attribute
cent = pd.DataFrame(G.degree(), columns='Person Edge_Count'.split())
cent.set_index('Person', inplace=True)
cent.sort_values('Edge_Count', ascending=False, inplace=True)
cent_max = cent.max().iloc[0]
cent['Degree'] = cent['Edge_Count']/cent_max

# cent = pd.DataFrame.from_dict(nx.degree_centrality(G), orient='index', columns='Degree'.split())
# cent["Betweeness"] = pd.DataFrame.from_dict(nx.betweenness_centrality(G), orient='index')
# cent["Eigenvector"] = pd.DataFrame.from_dict(nx.eigenvector_centrality(G), orient='index')
# cent = cent.sort_values(by='Degree', ascending=False)

# Add centrality information to node attribute
nx.set_node_attributes(G, cent['Degree'], 'centrality')
# Save centrality results
cent.to_csv(f"results/{re.sub('NMF_senti', 'centrality', FILE_NAME)}")


# --- Compute cliques and add clique group number as node attribute
# Construct a dictionary containing cliques within the network labeled by a clique#
networkCliques = {}
num = 0
for c in nx.find_cliques(G):
    networkCliques[num] = c
    num += 1
# Save clique results
pd.Series(networkCliques).to_csv(f"results/{re.sub('NMF_senti', 'cliques', FILE_NAME)}", header=False)

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

print('\nGraph construction complete!')


# ----- Save graph
graphFileName = re.sub('.csv', '.gpickle', FILE_NAME)
graphFileName = re.sub('NMF_senti', 'graph', graphFileName)
nx.write_gpickle(G, f"results/{graphFileName}")


# TODO: Construct normal graph with weighted edges using multigraph

# os.system('say Complete')