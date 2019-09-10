"""
@author: kaisoon
"""
# ----- Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import importlib
import drawGraph

DATE = '2017-12'
PATH = f"results/{DATE}/"
G = nx.read_gpickle(f"{PATH}ssm_{DATE}_weightedGraph.gpickle")
CGs = nx.read_gpickle(f"{PATH}ssm_{DATE}_cliqueGraphs.gpickle")
results = pd.read_csv(f"{PATH}ssm_{DATE}_results_NMF_senti.csv")


#%% Draw all Graphs
importlib.reload(drawGraph)
sns.set_style("white")
sns.set_context("talk")

FIG_SiZE = 4
LAYOUT = 'kamada'
groupings = 'party gender metro'.split()

for grp in groupings:
    fig = plt.figure(figsize=(FIG_SiZE * 3, FIG_SiZE * 2), dpi=300, tight_layout=True)
    drawGraph.draw(
        G,
        groupBy=grp,
        layout=LAYOUT,
        title=f"Same-sex Marriage Bill {DATE}",
        node_label=False,
    )
    fig.savefig(f"{PATH}figures/ssm_{DATE}_graph_by{grp.capitalize()}.png", dpi=300)
    plt.show()


#%% Draw all Clique Graphs
importlib.reload(drawGraph)
sns.set_style("white")
sns.set_context("talk")

# For an N by N subplot layout
N = 3
FIG_SiZE = N*2
subGraph = N * 110
LAYOUT = 'spring'
groupings = 'party gender metro'.split()

for grp in groupings:
    fig = plt.figure(figsize=(FIG_SiZE*3, FIG_SiZE*2), dpi=300, tight_layout=True)
    # Draw cliqueGraphs in subplots
    for k,gph in CGs.items():
        print(f"\nDrawing cliqueGraph{k}...")
        subGraph += 1
        plt.subplot(subGraph)
        drawGraph.draw(gph, groupBy=grp, layout=LAYOUT,
                       title=f"{DATE} Clique: {k}",
                       legend=False,
                       node_size=5,
                       font_size=6,
                       node_size_highCent= 10,
                       title_fontsize=12,
                       )
    fig.savefig(f"{PATH}figures/ssm_{DATE}_cliqueGraphs_by{grp.capitalize()}.png", dpi=300)
    plt.show()


#%% Distribution plots of coefMax, coeffDiff, sentiDiff, and centrality
sns.set_style("darkgrid")
sns.set_context("notebook")
FIG_SiZE = 3
LABEL_SIZE = 8

coeffMax = pd.read_csv(f"{PATH}distributions/ssm_{DATE}_topiccoeffMax.csv")
coeffDiff = pd.read_csv(f"{PATH}distributions/ssm_{DATE}_topicCoeffDiff.csv")
sentiDiff = pd.read_csv(f"{PATH}distributions/ssm_{DATE}_sentiDiff.csv")
cent = pd.read_csv(f"{PATH}ssm_{DATE}_centrality.csv")

fig = plt.figure(figsize=(FIG_SiZE*3, FIG_SiZE*2), dpi=300, tight_layout=True)
fig.suptitle(f"Same-sex Marriage Bill {DATE}", fontsize=12, y=0.99)

# ----- Distribution of coeffMax
plt.subplot(221)
ax = sns.distplot(coeffMax, kde=False)
# ax.set_ylim(0, 15)
ax.set_xlabel("Highest Topic Coefficient", fontsize=LABEL_SIZE)
ax.set_ylabel("Frequency", fontsize=LABEL_SIZE)

# ----- Distribution of coeffDiff
plt.subplot(222)
ax = sns.distplot(coeffDiff, kde=False)
# ax.set_ylim(0, 15)
ax.set_xlabel("Coefficient Difference of Top-two Topics", fontsize=LABEL_SIZE)
ax.set_ylabel(None)

# ----- Distribution of sentiDiff
plt.subplot(223)
ax = sns.distplot(sentiDiff['Senti_Diff'], kde=False)
# ax.set_ylim(0, 15)
ax.set_xlabel("Sentiment Intensity Difference", fontsize=LABEL_SIZE)
ax.set_ylabel("Frequency", fontsize=LABEL_SIZE)

# ----- Distribution of centrality
plt.subplot(224)
ax = sns.distplot(cent['Degree'], kde=False)
# ax.set_ylim(0, 15)
ax.set_xlabel("Centrality", fontsize=LABEL_SIZE)
ax.set_ylabel(None)

fig.savefig(f"{PATH}distributions/ssm_{DATE}_distributions.png", dpi=300)
fig.show()


#%% Template for main program
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import topicMod_NMF
import sentiAnalysis
import constructGraph
import constructCliqueGraph
import drawGraph

DATE = '2017-12'
PATH = f"results/{DATE}/"

# ----- Parameters for Topic Modelling
nTOPICS = 3
nWORDS = 15
COEFFMAX_PERC_THRES = 0.1
COEFFDIFF_PERC_THRES = 0.1
# ----- Parameters for Sentiment Analysis
SENTIDIFF_PERC_THRES = 0.2

# Load data
data = pd.read_csv(f"data/ssm_{DATE}_cleaned.csv")

#%% Topic Modeling
importlib.reload(topicMod_NMF)
results_nmf, topics_nmf, coeffMax, coeffDiff = topicMod_NMF.topicMod(data, nTOPICS, nWORDS, COEFFMAX_PERC_THRES, COEFFDIFF_PERC_THRES)

#%% Sentiment Analysis
importlib.reload(sentiAnalysis)
results_nmf_senti, sentiDiff_thres, sentiDiff = sentiAnalysis.sentiAn(results_nmf, SENTIDIFF_PERC_THRES)

#%% ----- Construct Graph
importlib.reload(constructGraph)
G, cent, cliques = constructGraph.constructG(results_nmf_senti, sentiDiff_thres)

#%% Draw Graph
importlib.reload(drawGraph)
FIG_SiZE = 4
LAYOUT = 'kamada'
GROUP = 'party'
fig = plt.figure(figsize=(FIG_SiZE * 3, FIG_SiZE * 2), dpi=300, tight_layout=True)
drawGraph.draw(
    G,
    groupBy=GROUP,
    layout=LAYOUT,
    title=f"Same-sex Marriage Bill {DATE}"
)
fig.savefig(f"{PATH}figures/ssm_{DATE}_graph_by{GROUP.capitalize()}.png", dpi=300)
plt.show()


#%% ----- Construct Clique Graphs
importlib.reload(constructCliqueGraph)
CGs = constructCliqueGraph.constructCG(G, cliques)

#%% Draw Clique Graphs
importlib.reload(drawGraph)
N = 3
FIG_SiZE = N*2
NODE_SIZE = 5
subGraph = N * 110
LAYOUT = 'spring'
fig = plt.figure(figsize=(FIG_SiZE*3, FIG_SiZE*2), dpi=300, tight_layout=True)
for k,gph in CGs.items():
    print(f"\nDrawing cliqueGraph{k}...")
    subGraph += 1
    plt.subplot(subGraph)
    drawGraph.draw(gph, groupBy=GROUP, layout=LAYOUT,
                   title=f"{DATE} Clique: {k}",
                   legend=False,
                   node_size=NODE_SIZE,
                   font_size=NODE_SIZE + 1,
                   node_size_highCent= NODE_SIZE*2,
                   title_fontsize=12,
                   )
fig.savefig(f"{PATH}figures/ssm_{DATE}_cliqueGraphs_by{GROUP}.png", dpi=300)
plt.show()


#%% Draw graphs with varying COEFFDIFF_PERC_THRES
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import topicMod_NMF
import sentiAnalysis
import constructGraph
import constructCliqueGraph
import drawGraph

importlib.reload(topicMod_NMF)
importlib.reload(sentiAnalysis)
importlib.reload(constructGraph)
importlib.reload(constructCliqueGraph)
importlib.reload(drawGraph)

DATE = '2017-12'
PATH = f"results/{DATE}/"

N = 3
FIG_SiZE = N*2
NODE_SIZE = 3
subGraph = N * 110
LAYOUT = 'kamada'
GROUP = 'party'
fig = plt.figure(figsize=(FIG_SiZE * 3, FIG_SiZE * 2), dpi=300, tight_layout=True)


coeffDiff_percThres_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for percThres in coeffDiff_percThres_list:
    # ----- Parameters for Topic Modelling
    nTOPICS = 3
    nWORDS = 15
    COEFFDIFF_PERC_THRES = percThres
    # ----- Parameters for Sentiment Analysis
    SENTIDIFF_PERC_THRES = 0.2

    # ----- Load data
    data = pd.read_csv(f"data/ssm_{DATE}_cleaned.csv")
    # ----- Topic Modeling
    results_nmf, topics_nmf, coeffDiff = topicMod_NMF.topicMod(data, nTOPICS, nWORDS, COEFFDIFF_PERC_THRES)
    # ----- Sentiment Analysis
    results_nmf_senti, sentiDiff_thres, sentiDiff = sentiAnalysis.sentiAn(results_nmf, SENTIDIFF_PERC_THRES)
    # ----- Construct Graph
    G, cent, cliques = constructGraph.constructG(results_nmf_senti, sentiDiff_thres)

    # ----- Draw Graph
    print(f"\nDrawing Graph with percthres={k}")
    subGraph += 1
    plt.subplot(subGraph)
    drawGraph.draw(
        G,
        groupBy=GROUP,
        layout=LAYOUT,
        title=f"{DATE} CoeffDiff PercThres: {percThres}",
        node_size=NODE_SIZE,
        font_size=NODE_SIZE + 1,
        node_size_highCent=NODE_SIZE*2,
        title_fontsize=12,
        legend=False,
    )

fig.savefig(f"{PATH}figures/ssm_{DATE}_graph_by{GROUP.capitalize()}_varyingCoeffDiff.png", dpi=300)
plt.show()
