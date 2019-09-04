#%% ----- Imports
FILE_NAME = "ssm_results_NMF_senti_2017-12.csv"
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import re


#%% ---------- Construct Network Graph
# If the difference in sentiment is higher than the threshold, both speeches do not agree with each other.
# TODO: Determine sentiDiff_thres statistically & empirically. Try significant levels of 0.01, 0.05, 0.1
# TODO: Find all sentiment difference between all speeches plot its distribution
sentiDiff_thres = 2

results = pd.read_csv(f"data/results/{FILE_NAME}")

G = nx.MultiGraph()
G.clear()

# ----- Add politicians as nodes in the graph
print('Adding nodes...')
for i in results.index:
    # Print progress...
    if i % 100 == 0:
        print(f"{i} of {len(results)}")

    row = results.loc[i]
    person = row['Person']
    # Only add actor if the actor hasn't already been added
    if not G.has_node(person):
        # Construct dataFrame for speechData attribute of node
        # Extract all results from the actor
        speechData = results[results['Person'] == person]
        speechData.index = range(len(speechData))

        # Drop unnecessary columns?
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


#%% ----- Draw Network
NODE_SIZE = 8
FIG_SiZE = 4
pal = pd.Series(sns.color_palette().as_hex(), index='blue orange green red purple brown pink grey yellow teal'.split())

# Sort actors into their parties such that they can be colour coded in the graph
party = pd.Series(nx.get_node_attributes(G, 'Party'))
partyNames = party.unique()
actors = pd.DataFrame()
for n in partyNames:
    actors = pd.concat([actors, pd.DataFrame(party[party == n].index, columns=n.split())], axis=1)

fig = plt.figure(figsize=(FIG_SiZE*3, FIG_SiZE*2), dpi=300, facecolor=pal['grey'], tight_layout=True)
plt.title(f"Same-sex Marriage Bill {FILE_NAME[-14:-4]}")

# Get position of nodes base on a layout
pos = nx.spring_layout(G, iterations=300, k=5)
# Increase k to spread nodes further apart
# pos = nx.kamada_kawai_layout(G)
# pos = nx.random_layout(G)

# TODO: Highlight node/s with high centrality
# ----- Draw nodes
nx.draw_networkx_nodes(G, pos, nodelist=list(actors['LP'].dropna()), node_size=NODE_SIZE*100, node_color=pal['blue'], alpha=0.7)
nx.draw_networkx_nodes(G, pos, nodelist=list(actors['ALP'].dropna()), node_size=NODE_SIZE*100, node_color=pal['red'], alpha=0.7)
nx.draw_networkx_nodes(G, pos, nodelist=list(actors['Nats'].dropna()), node_size=NODE_SIZE*100, node_color=pal['yellow'], alpha=0.7)
nx.draw_networkx_nodes(G, pos, nodelist=list(actors['AG'].dropna()), node_size=NODE_SIZE*100, node_color=pal['green'], alpha=0.7)
nx.draw_networkx_nodes(G, pos, nodelist=list(actors['IND'].dropna()), node_size=NODE_SIZE*100, node_color=pal['brown'], alpha=0.7)
nx.draw_networkx_nodes(G, pos, nodelist=list(actors['KAP'].dropna()), node_size=NODE_SIZE*100, node_color=pal['purple'], alpha=0.7)

# ----- Draw node labels
# Construct dict to reformat names from 'first last' to 'first\nlast'
nodes = G.nodes
nodeLabels = []
for n in nodes:
    nodeLabels.append(re.sub(' ', '\n', n))
nodeLabels = dict(zip(nodes, nodeLabels))

nx.draw_networkx_labels(G, pos, labels=nodeLabels, font_size=NODE_SIZE, font_color='black')
# Draw edges
nx.draw_networkx_edges(G, pos, width=0.25, edge_color=pal['grey'])

# Save figure
FORMAT = '.png'
graphFileName = re.sub('.csv', FORMAT, FILE_NAME)
graphFileName = re.sub('results', 'graph', graphFileName)

# fig.savefig(f"figures/{graphFileName}", dpi=300)
plt.show()


#%% Compute centrality
cent = pd.DataFrame(G.degree(), columns='Person Edge_Count'.split())
cent.set_index('Person', inplace=True)
cent.sort_values('Edge_Count', ascending=False, inplace=True)
cent_max = cent.max().iloc[0]
cent['Degree'] = cent['Edge_Count']/cent_max

# cent = pd.DataFrame.from_dict(nx.degree_centrality(G), orient='index', columns='Degree'.split())
# cent["Betweeness"] = pd.DataFrame.from_dict(nx.betweenness_centrality(G), orient='index')
# cent["Eigenvector"] = pd.DataFrame.from_dict(nx.eigenvector_centrality(G), orient='index')
# cent = cent.sort_values(by='Degree', ascending=False)

cent.to_csv(f"data/results/{re.sub('NMF_senti', 'centrality', FILE_NAME)}")
# TODO: Plot centrality of politicians against time


#%% Compute cliques
# TODO: Work out how to draw clique graphs
cliques = pd.DataFrame(list(nx.find_cliques(G))).transpose()
cliqueCnt = nx.graph_number_of_cliques(G)

cliques.to_csv(f"data/results/{re.sub('NMF_senti', 'cliques', FILE_NAME)}", index=False)

CG = nx.make_max_clique_graph(G)
nx.draw_networkx(CG)
plt.show()

# Save figure
FORMAT = '.png'
graphFileName = re.sub('.csv', FORMAT, FILE_NAME)
graphFileName = re.sub('results', 'cliques', graphFileName)
fig.savefig(f"figures/{graphFileName}", dpi=300)
plt.show()


#%% Add centrality and clique information as node attributes
# TODO: Add centrality and clique information as node attributes

i = results.iloc[10]
j = results.iloc[11]

data = pd.DataFrame([i, j])
speechData = pd.DataFrame([data['Speech_id'], data['Date']], axis=1)
speechData.index = range(len(data))


R = results
R1 = R.drop('Date Bill'.split(), axis=1)

S1 = nx.get_node_attributes(G, 'SpeechData')['Warren Entsch']
S2 = nx.get_edge_attributes(G, 'Topic')
print(S2)

#%% To get node and edge attributes from graph...
# To get node attributes
att = 'Party'
D1 = nx.get_node_attributes(G, att)
# get_node_attributes() returns a dict with node name as key and the attribute as value

# To get edge attributes
p1 = 'Warren Entsch'
p2 = 'Bill Shorten'
D2 = G.get_edge_data(p1, p2)
# MultiGraph.get_edge_data() returns a dict containing all edges drawn between two actors. De-referencing each edge returns another dict with the atribtue name as key and the its corresponding value
D2[0]
D2[0]['SpeechData']
