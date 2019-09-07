#%% ----- Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


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


#%%
# TODO: Write function to group edge weight into five classes of magnitude
import numpy as np
N = 5
# Extract all egdes with weights attribtes from graph
weights = nx.get_edge_attributes(G, 'weight')
# Compute weight relative to max weight
weight_relMax = {k : v/max(weights.values()) for (k, v) in weights.items()}

grouped = {}
grouped[0] = [k for k,v in weight_relMax.items() if v >= 0 and v < 0.2]
grouped[1] = [k for k,v in weight_relMax.items() if v >= 0.2 and v < 0.4]
grouped[2] = [k for k,v in weight_relMax.items() if v >= 0.4 and v < 0.6]
grouped[3] = [k for k,v in weight_relMax.items() if v >= 0.6 and v < 0.8]
grouped[4] = [k for k,v in weight_relMax.items() if v >= 0.8]


for k in range(5):
    print(len(grouped[k]))

S = pd.Series(weight_relMax).sort_values(ascending=False)
S = pd.Series(weights).sort_values(ascending=False)
