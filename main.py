"""
@author: kaisoon
"""
# ----- Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import importlib
import kaiGraph as kg

DATE = '2017-12'
PATH = f"results/{DATE}/"
G = nx.read_gpickle(f"{PATH}ssm_{DATE}_weightedGraph.gpickle")
CGs = nx.read_gpickle(f"{PATH}ssm_{DATE}_cliqueGraphs.gpickle")
results = pd.read_csv(f"{PATH}ssm_{DATE}_results_NMF_senti.csv")


#%% Draw all Graphs
importlib.reload(kg)
sns.set_style("white")
sns.set_context("talk")

FIG_SiZE = 4
LAYOUT = 'kamada'
groupings = 'party gender metro'.split()

for grp in groupings:
    fig = plt.figure(figsize=(FIG_SiZE * 3, FIG_SiZE * 2), dpi=300, tight_layout=True)
    kg.drawGraph(
        G,
        groupBy=grp,
        layout=LAYOUT,
        title=f"Same-sex Marriage Bill {DATE}"
    )
    fig.savefig(f"{PATH}figures/ssm_{DATE}_graph_by{grp.capitalize()}.png", dpi=300)
    plt.show()


#%% Draw all Clique Graphs
importlib.reload(kg)
sns.set_style("white")
sns.set_context("talk")

FIG_SiZE = 6
SUBGRAPH_LAYOUT = 331
LAYOUT = 'spring'
groupings = 'party gender metro'.split()

for grp in groupings:
    fig = plt.figure(figsize=(FIG_SiZE * 3, FIG_SiZE * 2), dpi=300, tight_layout=True)
    # Draw cliqueGraphs in subplots
    for k,gph in CGs.items():
        plt.subplot(SUBGRAPH_LAYOUT + k)
        kg.drawGraph(gph, groupBy=grp, layout=LAYOUT,
                      title=f"{DATE} Clique-{k}",
                      legend=False,
                      node_size=5,
                      font_size=6,
                      node_size_highCent= 10,
                      title_fontsize=12,
                     )
    fig.savefig(f"{PATH}figures/ssm_{DATE}_cliqueGraphs_by{grp.capitalize()}.png", dpi=300)
    plt.show()


#%% Distribution plots of coeffDiff, sentiDiff, and centrality
sns.set_style("darkgrid")
sns.set_context("notebook")
LABEL_SIZE = 9

coeffDiff = pd.read_csv(f"{PATH}distributions/ssm_{DATE}_topicCoeffDiff.csv")
sentiDiff = pd.read_csv(f"{PATH}distributions/ssm_{DATE}_sentiDiff.csv")
cent = pd.read_csv(f"{PATH}ssm_{DATE}_centrality.csv")

fig = plt.figure(figsize=(12, 8), dpi=300, tight_layout=True)

# ----- Distribution of coeffDiff of the top-two topics
plt.subplot(231)
ax = sns.distplot(coeffDiff['Topic_Coeff_Diff'], kde=False)
# ax.set_ylim(0, 15)
ax.set_xlabel("Topic Coefficient Difference", fontsize=LABEL_SIZE)
ax.set_ylabel("Frequency", fontsize=LABEL_SIZE)

plt.subplot(234)
ax = sns.distplot(coeffDiff['Percentile'], kde=False)
# ax.set_ylim(0, 15)
ax.set_xlabel("Topic Coefficient Difference Percentile", fontsize=LABEL_SIZE)
ax.set_ylabel("Frequency", fontsize=LABEL_SIZE)


# ----- Distribution of sentiDiff of speech on the same topic
plt.subplot(232)
ax = sns.distplot(sentiDiff['Senti_Diff'], kde=False)
# ax.set_ylim(0, 15)
ax.set_title(f"Same-sex Marriage Bill {DATE}")
ax.set_xlabel("Sentiment Intensity Difference", fontsize=LABEL_SIZE)
ax.set_ylabel(None)

plt.subplot(235)
ax = sns.distplot(sentiDiff['Percentile'], kde=False)
# ax.set_ylim(0, 15)
ax.set_xlabel("Sentiment Intensity Difference Percentile", fontsize=LABEL_SIZE)
ax.set_ylabel(None)


# ----- Distribution of centrality
plt.subplot(233)
ax = sns.distplot(cent['Degree'], kde=False)
# ax.set_ylim(0, 15)
ax.set_xlabel("Centrality", fontsize=LABEL_SIZE)
ax.set_ylabel(None)

plt.subplot(236)
ax = sns.distplot(cent['Percentile'], kde=False)
# ax.set_ylim(0, 15)
ax.set_xlabel("Centrality Percentile", fontsize=LABEL_SIZE)
ax.set_ylabel(None)


fig.savefig(f"{PATH}distributions/ssm_{DATE}_distributions.png", dpi=300)
fig.show()


#%%