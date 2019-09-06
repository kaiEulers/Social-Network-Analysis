#%% ----- Imports
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import re


#%% THIS IS HOW YOU APPEND ROW TO DATAFRAMES!!
df = pd.DataFrame(np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
    ]),
    columns='A B C'.split()
)

s = pd.Series([10, 11, 12], index=df.columns)
df = df.append(s, ignore_index=True)
df.dtypes


#%% THIS IS HOW YOU INTERATE THROUGH DATAFRAMES!!
df = pd.DataFrame(np.array([
    'a b c'.split(),
    'e f g'.split(),
    'h i j'.split()
    ]),
    columns='col1 col2 col3'.split()
)

for index,row in df.iterrows():
    print(index)
    print(row, '\n')


#%% Change datatype of dataframes
df = df.astype('object')
df.dtypes

df.iloc[0]['A'] = [1,2,3]


#%% MultiGraph
MG = nx.MultiGraph()
MG.clear()

MG.add_node('a')
MG.add_node('b')
MG.add_node('c')

MG. add_edge('a', 'b')
MG. add_edge('a', 'b')
MG. add_edge('a', 'c')
# MG.add_weighted_edges_from([
#     ('a', 'b', 1),
#     ('a', 'b', 1),
#     ('a', 'c', 1)
# ])

MG.degree()
nx.degree_centrality(MG)

pos = nx.spring_layout(MG)
nx.draw_networkx(MG)
plt.show()


#%% Re-write node attribute
G = nx.Graph()
G.clear()

G.add_node(0, time=5)
G.node.data()
G.add_node(0, time=+1)
G.nodes.data()

# USE G.node.data() TO EXTRACT ALL NODES AND ITS DATA AS A DICTIONARY!!!
D = G.nodes.data()
D[0]['time']

nx.draw_networkx(G)
plt.show()


#%% Data Quantities
data = pd.read_csv(f"/Users/kaisoon/Google Drive/Code/Python/COMP90055_project/data/ssm_cleaned.csv")
data_2017_12 = pd.read_csv(f"/Users/kaisoon/Google Drive/Code/Python/COMP90055_project/data/ssm_cleaned_2017-12.csv")

# Quantities
# Origianl dataset
data.shape
# Dataset from 04-12-2017 to 07-12-2017
data_2017_12.shape
# Dataset after topic modelling filter
results.shape

# Percentage of 12-2017 dataset
data_2017_12.shape[0]/data.shape[0]
# Percentage of dataset after topic modelling filter
results.shape[0]/data_2017_12.shape[0]

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
