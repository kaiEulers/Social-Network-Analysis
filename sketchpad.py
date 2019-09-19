#%% ----- Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import importlib
import group

DATE = '2017-12'
FIG_SiZE = 4
SUBGRAPH_LAYOUT = 221
PATH = f"results/{DATE}/"
data = pd.read_csv(f"{PATH}ssm_{DATE}_results_NMF_senti.csv")
G = nx.read_gpickle(f"{PATH}ssm_{DATE}_weightedGraph.gpickle")
CGs = nx.read_gpickle(f"{PATH}ssm_{DATE}_weightedGraph.gpickle")

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



