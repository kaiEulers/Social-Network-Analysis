#%% ----- Imports
import time as tm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import importlib

DATE = '2017'
RES = 'para'
METHOD = 'nmf'
PATH = f"results/resolution_{RES}/"

FIG_SiZE = 4
SUBGRAPH_LAYOUT = 221

data = pd.read_csv(f"data/ssm_rel.csv")
results = pd.read_csv(f"{PATH}ssm_results_{METHOD}_senti.csv")
G = nx.read_gpickle(f"{PATH}{DATE}/ssm_weightedGraph_{DATE}.gpickle")
# CGs = nx.read_gpickle(f"{PATH}ssm_cliqueGraph.gpickle")

coeffPrps = pd.read_csv(f"{PATH}stats/ssm_topicCoeffPrps_{METHOD}.csv", index_col=0)
sentiDiff = pd.read_csv(f"{PATH}stats/ssm_sentiDiff_{METHOD}.csv", index_col=0)



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
nx.draw_networkx(
    TG,
    # THIS IS HOW YOU COLOUR NODES WITH DIFFERENT CLQ_PROFILE_COLOURS!!!!!
    node_color='r g b'.split(),
    edge_color='magenta teal yellow'.split(),
    linewidths=3, edgecolors='black',
)
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

# USE TG.node.text() TO EXTRACT ALL NODES AND ITS text AS A DICTIONARY!!!
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



#%% TODO: Audit speeches with the strongest topic for each time-frame
from kaiFunctions import load_pickle
RES = 'para'
nTOPICS = 11
results_dict = load_pickle(f"{PATH}ssm_resultsDict_{RES}Res_{nTOPICS}topics")

tf = '2004'
topic = 7
R = results_dict[tf]
R7 = R[R['Topic'] == topic]
R7_coeffMax = R7.nlargest(5, 'CoeffMax')
print(R7_coeffMax.iloc[0]['Speech'])


#%% Sort actorProfiles by attribute
actorProfile_dict = load_pickle(f"{PATH}ssm_actorProfileDict_{nTOPICS}topics_{METHOD}")

actorProfileSorted_dict = {}
for tf in TF_DATES:
    actorProfileSorted_dict[tf] = actorProfile_dict[tf].sort_values(['BtwnCentrality'], ascending=False)
    actorProfileSorted_dict[tf].to_csv(f"{PATH}ssm_actorProfile_{RES}Res_{nTOPICS}topics_{tf}.csv")


#%% TODO: Audit speeches of actors with high betweenness centrality
results_dict = load_pickle(f"{PATH}ssm_resultsDict_{nTOPICS}topics_{METHOD}")
actorProfile_dict = load_pickle(f"{PATH}ssm_actorProfileDict_{nTOPICS}topics_{METHOD}")

nSPEECHES = 5
nACTORS = 4

for tf in TF_DATES:
    # Retrieve top-nACTORS with highest centrality
    ap = actorProfile_dict[tf].sort_values('BtwnCentrality', ascending=False)[:nACTORS]
    # Remove irrelevant topics
    results = results_dict[tf]
    results = results[results['Topic'] != -1]
    for name in ap.index:
        oneActor = results[results['Person'] == name]
        oneActor = oneActor.sort_values('CoeffMax', ascending=False)
        for i, (_, row) in enumerate(oneActor.iloc[:nSPEECHES].iterrows()):
            row.to_csv(f"realityChecks/speeech-export/{tf}/{name}{i}_{tf}.csv", header=False)

#%% TODO: Audit speeches of particular actors
results_dict = load_pickle(f"{PATH}ssm_resultsDict_{nTOPICS}topics_{METHOD}")

name = 'Tony Abbott'
tf = 'All'
nSPEECHES = 5

results = results_dict[tf]
results = results[results['Topic'] != -1]

oneActor = results[results['Person'].str.contains(name)]
oneActor = oneActor.sort_values('CoeffMax', ascending=False)

for i, (_, row) in enumerate(oneActor.iloc[:nSPEECHES].iterrows()):
    row.to_csv(f"realityChecks/speeech-export/{name}{i}_{tf}.csv", header=False)



#%% Compute percentage of speeches used in each timeframe
results_dict = load_pickle(f"{PATH}ssm_resultsDict_{nTOPICS}topics_{METHOD}")

speechPerc = pd.DataFrame(columns='Percentage'.split())
for tf in results_dict:
    speechPerc.loc[tf] = len(results_dict[tf])/len(results_dict['All'])
    print(f"% of {tf}: {round(speechPerc['Percentage'].loc[tf]*100, 2)}%")
speechPerc.to_csv(f"{PATH}csv-export/ssm_speechPerc_{nTOPICS}topics_{tf}.csv")

#%% Attribute composition
actorProfile_dict = load_pickle(f"{PATH}ssm_actorProfileDict_{nTOPICS}topics_{METHOD}")
tf = 'All'

comp = []
for tf in TIME_FRAMES['termsPM']:
    ap = actorProfile_dict[tf]

    male = ap[ap['Gender'] == 'Male']
    female = ap[ap['Gender'] == 'Female']
    coalMale = male[male['PartyAgg'] == 'Coalition']
    coalFemale = female[female['PartyAgg'] == 'Coalition']
    laborMale = male[male['PartyAgg'] == 'Labor']
    laborFemale = female[female['PartyAgg'] == 'Labor']

    coal = ap[ap['PartyAgg'] == 'Coalition']
    labor = ap[ap['PartyAgg'] == 'Labor']
    greens = ap[ap['PartyAgg'] == 'Greens']
    minor = ap[ap['PartyAgg'] == 'Minor Party']

    row = [
        len(male)/len(ap),
        len(female)/len(ap),
        len(coalMale)/len(ap),
        len(coalFemale)/len(ap),
        len(laborMale)/len(ap),
        len(laborFemale)/len(ap),
        len(coal)/len(ap),
        len(labor)/len(ap),
        len(greens)/len(ap),
        len(minor)/len(ap),
    ]
    comp.append(row)

comp = pd.DataFrame(
    comp,
    columns='Male Female CoalMale CoalFemale LaborMale LaborFemale Coal Labor Greens Minor'.split(),
    index=TIME_FRAMES['termsPM']
)
comp.to_csv(f"{PATH}csv-export/composition_{nTOPICS}topics.csv")


#%% Examine sentiment of results
FIG_SIZE = 3
senti_pos = results[results['Senti_comp'] > 0]['Senti_comp']
print(senti_pos.describe())
senti_neg = results[results['Senti_comp'] < 0]['Senti_comp']
print(senti_neg.describe())

fig = plt.figure(figsize=(FIG_SIZE*2, FIG_SIZE*2), dpi=300)

ax = plt.subplot(211)
sns.distplot(senti_pos, kde=False)
ax.axvline(x=senti_pos.mean(), c=cp.cbPaired['red'])
ax.set_xlabel("Positive Sentiment")

ax = plt.subplot(212)
sns.distplot(senti_neg, kde=False)
ax.set_xlabel("Negative Sentiment")
ax.axvline(x=senti_neg.mean(), c=cp.cbPaired['red'])
plt.tight_layout()
plt.show()


#%% Test Jaccard & Cosine Similarity
import random as rand
import similarity as simi
importlib.reload(simi)

actorResults = pd.read_csv(f"results/resolution_para/ssm_topicProfiles_paraRes_11topics_ALL.csv")
L = actorResults.index.tolist()
p1, p2 = rand.choices(L, k=2)

# p1 = 'Kevin Hogan'
# p2 = 'Brian Mitchell'

t1 = actorResults.loc[p1]['Topics']
t1 = t1.split()
t2 = actorResults.loc[p2]['Topics']
t2 = t2.split()

v1 = actorResults.loc[p1]['TopicVector']
v1 = list(map(int, v1.split()))
v2 = actorResults.loc[p2]['TopicVector']
v2 = list(map(int, v2.split()))

print(f"{p1:{20}}{t1}")
print(f"{p2:{20}}{t2}")
print(f"Jaccard:\t{simi.jaccard(t1, t2)}")
print(f"Cosine:\t{simi.cosine(v1, v2)}")


#%%
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))

# Load the example car crash dataset
crashes = sns.load_dataset("car_crashes").sort_values("total", ascending=False)

# Plot the total crashes
sns.set_color_codes("pastel")
sns.barplot(x="total", y="abbrev", data=crashes,
            label="Total", color="b")

# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
sns.barplot(x="alcohol", y="abbrev", data=crashes,
            label="Alcohol-involved", color="b")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 24), ylabel="",
       xlabel="Automobile collisions per billion miles")
sns.despine(left=True, bottom=True)
plt.show()



#%%
import kaiFunctions as kf
from importlib import reload
import re
reload(kf)

L = [1, 2, 3, 4, 5]
S = '1 2 3 4 5'
S_float = '1.1. 2.2 3.3 4.4. 5.5'

re.findall('\.', S_float.split()[0])
notNumList = [re.findall('\D', string) for string in S_float.split()]
[re.findall('\\.', notNum)for notNum in notNumList]

kf.parser(L)


#%%
