"""
@author: kaisoon
"""
# %% Template for main program
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import importlib
import topicMod
import sentiAnalysis
import constructGraph
import constructCliqueGraph
import drawGraph
import constructMultiGraph

# DATE = '2017-12'
PATH = f"results/"
TIME_FRAMES = '2004 2012 2013 2015 2016 2017'.split()
data = pd.read_csv(f"data/ssm_rel.csv")

# %% ----- Topic Modelling
importlib.reload(topicMod)
# Parameters for Topic Modelling
nTOPICS = 4
nWORDS = 10
COEFFMAX_PERC_THRES = 0.1
COEFFDIFF_PERC_THRES = 0.1
# Select between 'nmf' or 'lda' topic modelling
METHOD = 'nmf'

results_topicMod, topics, coeffMax, coeffDiff = topicMod.model(data, nTOPICS, nWORDS, COEFFMAX_PERC_THRES, COEFFDIFF_PERC_THRES, method=METHOD)
len(results_topicMod)

# Save results
results_topicMod.to_csv(f"{PATH}ssm_results_{METHOD}.csv", index=False)
topics.to_csv(f"{PATH}ssm_topics_{METHOD}.csv")
coeffMax.to_csv(f"{PATH}stats/ssm_topicCoeffMax_{METHOD}.csv")
coeffDiff.to_csv(f"{PATH}stats/ssm_topicCoeffDiff_{METHOD}.csv")

# Extract speeches with the highest coeffMax to determine topic name
speech_id = topics.loc['speechId']
highestCoeffMax = results_topicMod[results_topicMod['Speech_id'].isin(speech_id)]
highestCoeffMax = highestCoeffMax.sort_values(by='Topic')

# Delete existing saved speeches
for i in range(nTOPICS):
    if os.path.isfile(f"{PATH}realityChecks/highestCoeffMax_{METHOD}/highestCoeffMax_{METHOD}_topic{i}.csv"):
        os.remove(f"{PATH}realityChecks/highestCoeffMax_{METHOD}/highestCoeffMax_{METHOD}_topic{i}.csv")
# Save each speech
for i, row in highestCoeffMax.iterrows():
    row_T = row.transpose()
    row_T.to_csv(f"{PATH}realityChecks/highestCoeffMax_{METHOD}/highestCoeffMax_{METHOD}_topic{row['Topic']}.csv", header=True)


# %% ----- Sentiment Analysis
importlib.reload(sentiAnalysis)
# Parameters for Sentiment Analysis
SENTIDIFF_PERC_THRES = 0.1

results_topicMod_senti, sentiDiff_thres, sentiDiff = sentiAnalysis.sentiAnal(results_topicMod, SENTIDIFF_PERC_THRES)
len(results_topicMod_senti)

# Save results
results_topicMod_senti.to_csv(f"{PATH}ssm_results_{METHOD}_senti.csv", index=False)
sentiDiff.to_csv(f"{PATH}stats/ssm_sentiDiff_{METHOD}.csv")
with open(f"{PATH}ssm_sentiDiffThres_{METHOD}.txt", "w") as file:
    file.write(str(sentiDiff_thres))


# %% ----- Construct Graph
importlib.reload(constructGraph)

G_dict = {}
centDict = {}
cliquesDict = {}

for k in TIME_FRAMES:
    results = pd.read_csv(f"{PATH}{k}/ssm_results_{k}.csv")
    G_dict[k], centDict[k], cliquesDict[k] = constructGraph.constructG(results, sentiDiff_thres)

    # Save graph
    nx.write_gpickle(G_dict[k], f"{PATH}{k}/ssm_weightedGraph_{k}.gpickle")
    # Save centrality results
    centDict[k].to_csv(f"{PATH}{k}/ssm_centrality_{k}.csv")
    # Save clique results
    with open(f"{PATH}{k}/ssm_cliques_{k}.pickle", "wb") as file:
        pickle.dump(cliquesDict[k], file)


# %% ----- Draw Graph
importlib.reload(drawGraph)
FIG_SIZE = 4
LAYOUT = 'kamada'
GROUP = 'party'

# TODO: Continue from here! Plot 3 sets of 6 graphs from the 6 timeframes.
figGraph = plt.figure(figsize=(FIG_SIZE*3, FIG_SIZE*2), dpi=300, tight_layout=True)
drawGraph.draw(
    G,
    groupBy=GROUP,
    layout=LAYOUT,
    title=f"Same-sex Marriage Bill {DATE}"
)
figGraph.savefig(f"{PATH}figures/ssm_{DATE}_graph_by{GROUP.capitalize()}.png", dpi=300)
plt.show()

# %% ----- Construct Clique Graphs
importlib.reload(constructCliqueGraph)
CGs = constructCliqueGraph.constructCG(G, cliques)
len(CGs)

# %% ----- Draw Clique Graphs
importlib.reload(drawGraph)
# For an N by N subplot layout
if len(CGs) <= 4:
    N = 2
else:
    N = 3
FIG_SIZE = N*2
NODE_SIZE = 5
subGraph = N*110
LAYOUT = 'spring'
figGraph = plt.figure(figsize=(FIG_SIZE*3, FIG_SIZE*2), dpi=300, tight_layout=True)
for k, gph in CGs.items():
    print(f"\nDrawing cliqueGraph{k}...")
    subGraph += 1
    plt.subplot(subGraph)
    drawGraph.draw(gph, groupBy=GROUP, layout=LAYOUT,
                   title=f"{DATE} Clique: {k}",
                   legend=False,
                   node_size=NODE_SIZE,
                   font_size=NODE_SIZE + 1,
                   title_fontsize=12,
                   )
figGraph.savefig(f"{PATH}figures/ssm_{DATE}_cliqueGraphs_by{GROUP}.png", dpi=300)
plt.show()

# %% ----- Construct Multi-graph and compute topic count
importlib.reload(constructMultiGraph)
MG, topicCount = constructMultiGraph.constructMG(results_topicMod_senti, sentiDiff_thres)

# %% Draw all Graphs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import importlib
import drawGraph

importlib.reload(drawGraph)
sns.set_style("white")
sns.set_context("talk")

DATE = '2017-12'
PATH = f"results/{DATE}/"
G = nx.read_gpickle(f"{PATH}ssm_{DATE}_weightedGraph.gpickle")
CGs = nx.read_gpickle(f"{PATH}ssm_{DATE}_cliqueGraphs.gpickle")
data = pd.read_csv(f"{PATH}ssm_{DATE}_results_topicMod_senti.csv")

FIG_SIZE = 4
LAYOUT = 'kamada'
# LAYOUT = 'spring'
groupings = 'party gender metro'.split()

for grp in groupings:
    fig = plt.figure(figsize=(FIG_SIZE*3, FIG_SIZE*2), dpi=300, tight_layout=True)
    drawGraph.draw(
        G,
        groupBy=grp,
        layout=LAYOUT,
        title=f"Same-sex Marriage Bill {DATE}",
    )
    fig.savefig(f"{PATH}figures/ssm_{DATE}_graph_by{grp.capitalize()}.png", dpi=300)
    plt.show()

# %% Draw all Clique Graphs
importlib.reload(drawGraph)
sns.set_style("white")
sns.set_context("talk")

# For an N by N subplot layout
if len(CGs) <= 4:
    N = 2
else:
    N = 3
FIG_SIZE = N*2

LAYOUT = 'spring'
groupings = 'party gender metro'.split()

for grp in groupings:
    fig = plt.figure(figsize=(FIG_SIZE*3, FIG_SIZE*2), dpi=300, tight_layout=True)
    subGraph = N*110
    # Draw cliqueGraphs in subplots
    for k, gph in CGs.items():
        print(f"\nDrawing cliqueGraph{k}...")
        subGraph += 1
        plt.subplot(subGraph)
        drawGraph.draw(gph, groupBy=grp, layout=LAYOUT,
                       title=f"{DATE} Clique: {k}",
                       legend=False,
                       node_size=5,
                       font_size=6,
                       title_fontsize=12,
                       )
    fig.savefig(f"{PATH}figures/ssm_{DATE}_cliqueGraphs_by{grp.capitalize()}.png", dpi=300)
    plt.show()

# %% Draw graphs with varying COEFFDIFF_PERC_THRES
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import topicMod
import sentiAnalysis
import constructGraph
import constructCliqueGraph
import drawGraph

importlib.reload(topicMod)
importlib.reload(sentiAnalysis)
importlib.reload(constructGraph)
importlib.reload(constructCliqueGraph)
importlib.reload(drawGraph)

DATE = '2017-12'
PATH = f"results/varying_coeffDiff_percthres{DATE}/"

N = 3
FIG_SIZE = N*2
NODE_SIZE = 3
subGraph = N*110
LAYOUT = 'kamada'
GROUP = 'party'
fig = plt.figure(figsize=(FIG_SIZE*3, FIG_SIZE*2), dpi=300, tight_layout=True)

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
    results_topicMod, topics, coeffDiff = topicMod.model(data, nTOPICS, nWORDS,
                                                         COEFFDIFF_PERC_THRES)
    # ----- Sentiment Analysis
    results_topicMod_senti, sentiDiff_thres, sentiDiff = sentiAnalysis.sentiAn(results_topicMod,
                                                                               SENTIDIFF_PERC_THRES)
    # ----- Construct Graph
    G, cent, cliques = constructGraph.constructG(results_topicMod_senti, sentiDiff_thres)

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

fig.savefig(f"{PATH}figures/ssm_{DATE}_graph_by{GROUP.capitalize()}_varyingCoeffDiff.png",
            dpi=300)
plt.show()
