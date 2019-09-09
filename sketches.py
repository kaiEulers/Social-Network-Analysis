#%% ----- Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import importlib
import group
import group
import kaiGraph as kg

DATE = '2017-12'
FIG_SiZE = 4
SUBGRAPH_LAYOUT = 221
PATH = f"results/{DATE}/"
G = nx.read_gpickle(f"{PATH}ssm_{DATE}_weightedGraph.gpickle")
CGs = nx.read_gpickle(f"{PATH}ssm_{DATE}_weightedGraph.gpickle")


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

df.drop(0)

# # Use df.iterrows() to return an iterator that iterates through rows
# for index,row in df.iterrows():
#     print(index)
#     print(row)

# # Use df.iterritems() to return an iterator that iterates through columns
# for index,col in df.iteritems():
#     print(index)
#     print(col, '\n')

# Use df.itertuples() to return an iterator that iterates through rows as tuples
for i in df.itertuples():
    print(i)


#%% Change datatype of dataframes
df = df.astype('object')
df.dtypes

df.iloc[0]['A'] = [1,2,3]


#%% Graph
TG = nx.Graph()
TG.clear()

TG.add_node('a')
TG.add_node('b')
TG.add_node('c')

TG.add_edge('a', 'b', colour='red', n=1)
TG.add_edge('b', 'c', colour='green', n=2)
TG.add_edge('a', 'c', colour='blue', n=3)

# To get all attributes from an edge
edgeData = TG.get_edge_data('a', 'c')

# To get one attribute from all edges
nx.get_edge_attributes(TG, 'colour')

TG.add_edge('a', 'c', colour='pink')
print(TG.edges.data())

pos = nx.spring_layout(TG)
nx.draw_networkx(TG)
plt.show()

# To get all edges adjacent to a node
TG.edges('a')



#%%
D1 = {'a': 1, 'b': 2, 'c': 3}
D2 = {'d': 4, 'e': 5, 'f': 6}

D = [D1, D2]





#%% Re-write node attribute
TG = nx.Graph()
TG.clear()

TG.add_node(0, time=5)
TG.node.data()
TG.add_node(0, time=+1)
TG.nodes.data()

# USE TG.node.data() TO EXTRACT ALL NODES AND ITS data AS A DICTIONARY!!!
D = TG.nodes.data()
D[0]['time']

nx.draw_networkx(TG)
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
D1 = nx.get_node_attributes(TG, att)
# get_node_attributes() returns a dict with node name as key and the attribute as value

# To get edge attributes
p1 = 'Warren Entsch'
p2 = 'Bill Shorten'
D2 = TG.get_edge_data(p1, p2)
# MultiGraph.get_edge_data() returns a dict containing all edges drawn between two actors. De-referencing each edge returns another dict with the atribtue name as key and the its corresponding value


#%% THIS IS HOW YOU SHOULD DRAW MULTIPLE CLIQUE GRAPHS!!!
import networkx as nx
import random as rand
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
sns.set_context("notebook")
DATE = '12-2017'

nodes = {k: list(v.nodes) for k,v in CGs.items()}
nodes1 = {k: rand.choices(v, k=rand.randint(1, len(v))) for k,v in nodes.items()}
nodes2 = {}
for k in range(len(nodes)):
    nodes2[k] = [v for v in nodes[k] if v not in nodes1[k]]

print(f"Total # Nodes\t{[len(v) for k,v in nodes.items()]}")
print(f"# in Nodes1 \t{[len(v) for k,v in nodes1.items()]}")
print(f"# in Nodes2 \t{[len(v) for k,v in nodes2.items()]}")

edges = {k: list(v.edges) for k,v in CGs.items()}
[len(v) for k,v in edges.items()]
edges1 = {k: rand.choices(v, k=rand.randint(1, len(v))) for k,v in edges.items()}
[len(v) for k,v in edges1.items()]
edges2 = {}
for k in range(len(edges)):
    edges2[k] = [v for v in edges[k] if v not in edges1[k]]
[len(v) for k,v in edges2.items()]

print(f"\nTotal # Edges\t{[len(v) for k,v in edges.items()]}")
print(f"# in Edges1 \t{[len(v) for k,v in edges1.items()]}")
print(f"# in Edges2 \t{[len(v) for k,v in edges2.items()]}")


TGs = {}
TG = nx.Graph()
for k,n in nodes.items():
    TG.clear()
    TG.add_nodes_from(n)
    TGs[k] = TG

fig = plt.figure(figsize=(12, 8), dpi=300)
for k,v in CGs.items():

    plt.subplot(221+k)
    plt.title(f"Clique {k}")

    # pos = nx.kamada_kawai_layout(v, weight=None)
    pos = nx.spring_layout(v)
    nx.draw_networkx_nodes(v, pos, nodelist=nodes1[k], node_color='red' ,node_size=200, font_size=9, width=0.25)
    nx.draw_networkx_nodes(v, pos, nodelist=nodes2[k], node_color='green' ,node_size=200, font_size=9, width=0.25)

    # nx.draw_networkx(v, pos, nodelist=nodes1[k], node_color='red' ,node_size=200, font_size=9, width=0.25)
    # nx.draw_networkx(v, pos, nodelist=nodes1[k], node_color='blue' , node_size=200, font_size=9, width=0.25)'

plt.show()

#%% Attempted to sort nodes by centrality and party
nodes = dict(G.node.data())

p = 'Bill Shorten'
nodes[p]

parties = nx.get_node_attributes(G, 'parties')
party_unique = list(set(parties.values()))
party_unique.sort()
party_fullName = [
    "Australian Greens",
    "Australian Labor Party",
    "Independent",
    "Katter's Australia Party",
    "Liberal Party",
    "Nationals Party of Australia",
]
leg = zip(party_unique, party_fullName)


#%% Test Graph
import random as rand
S = "a b c d e f g".split()
nodes = rand.choices(S, k=rand.randint(1, len(S)))
nodes
TG = nx.Graph()
TG.clear()
TG.add_nodes_from(nodes)

pos = nx.kamada_kawai_layout(TG, weight=None)
# nx.draw_networkx_nodes(TG, pos, nodelist=nodes, label='blue')
nx.draw_networkx_nodes(TG, pos, nodelist=nodes, label='blue')
nx.draw_networkx_nodes(TG, pos, nodelist=rand.choice(nodes), linewidths=3, edgecolors='black')
plt.legend()
plt.show()


#%%
