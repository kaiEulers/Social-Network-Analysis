"""
@author: kaisoon
"""
# %% Template for main program
import os
import pickle
from datetime import datetime as dt
import pandas as pd, numpy as np
import seaborn as sns, matplotlib.pyplot as plt
import networkx as nx
import importlib
import topicMod, sentiAnalysis, constructGraph, constructCliqueGraph, drawGraph, constructMultiGraph
import miscFuncs as mf

# DATE = '2017-12'
PATH = f"results/"
TIME_FRAMES = '2004 2012 2013 2015 2016 2017'.split()
data = pd.read_csv(f"data/ssm_rel_lemma.csv")


# %% ----- Topic Modelling
importlib.reload(topicMod)
# Parameters for Topic Modelling
nTOPICS = 6
nWORDS = 10
COEFFMAX_STD = -1.5
COEFFDIFF_STD = -1.5
# Select between 'nmf' or 'lda' topic modelling
METHOD = 'nmf'

results_topicMod, topics, coeffPrps_thres, coeffPrps = topicMod.model(data, nTOPICS, nWORDS, COEFFMAX_STD, COEFFDIFF_STD, METHOD=METHOD, TRANSFORM_METHOD='box-cox' ,PLOT=True)

# Save results
results_topicMod.to_csv(f"{PATH}ssm_results_{METHOD}.csv", index=False)
topics.to_csv(f"{PATH}ssm_{nTOPICS}topics_{METHOD}.csv")
coeffPrps_thres.to_csv(f"{PATH}ssm_thresholds_{METHOD}.csv", header=False)
coeffPrps.to_csv(f"{PATH}/stats/ssm_topicCoeffPrps_{METHOD}.csv")

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


#%% ----- Name Topics


# %% ----- Sentiment Analysis
importlib.reload(sentiAnalysis)
# Parameters for Sentiment Analysis
TAIL_THRES = 0.1
SENTIDIFF_STD = 1.5

results_topicMod_senti, sentiDiff_thres, sentiDiff = sentiAnalysis.analyse(results_topicMod, TAIL_THRES, SENTIDIFF_STD, PLOT=True)
len(results_topicMod_senti)

# Save results
results_topicMod_senti.to_csv(f"{PATH}ssm_results_{METHOD}_senti.csv", index=False)
# TODO: Load "ssm_thresholds_nmf.csv", concat sentiDiff, and save it again
sentiDiff.to_csv(f"{PATH}stats/ssm_sentiDiff_{METHOD}.csv")
with open(f"{PATH}ssm_sentiDiffThres_{METHOD}.txt", "w") as file:
    file.write(str(sentiDiff_thres))


#%% Divide results into time-frames
importlib.reload(mf)

results = pd.read_csv("results/ssm_results_nmf_senti.csv")
results = mf.convert2datetime(results)

# ---------------------------------------------------------------------------
# 2004: Bills from 2004-05-27 to 2004-06-24
# Howard amended Marriage Act 1961 to be defined as a "union of a man and a woman to the exclusion of all others"
# ---------------------------------------------------------------------------
# 2012: Bills from 2012-02-13 to 2012-09-10
# First time House of Reps declared their position on SSM
# ---------------------------------------------------------------------------
# 2013: Bills from 2013-03-18 to 2013-06-24
# New Zealand legislated for SSM on the 17th April
# ---------------------------------------------------------------------------
# 2015: Bills from 2015-06-01 to 2015-11-23
# US legalised nation-wide SSM on the 26th June
# ---------------------------------------------------------------------------
# 2016: Bills from 2016-02-08 to 2016-11-21
# SSM plebiscite done between 12th Sept and 7th Nov
# ---------------------------------------------------------------------------
# 2017: Bills from 2017-09-13 to 2017-12-07
# Plebiscite results released on 15th Nov. SSM law passed on 7th Dec
# ---------------------------------------------------------------------------
resultsDict = {}
resultsDict['2004'] = results[(results['Date'] >= dt(2004, 5, 27)) & (results['Date'] <= dt(2004, 6, 24))]
resultsDict['2012'] = results[(results['Date'] >= dt(2012, 2, 13)) & (results['Date'] <= dt(2012, 9, 10))]
resultsDict['2013'] = results[(results['Date'] >= dt(2013, 3, 18)) & (results['Date'] <= dt(2013, 6, 24))]
resultsDict['2015'] = results[(results['Date'] >= dt(2015, 6, 1)) & (results['Date'] <= dt(2015, 11, 23))]
resultsDict['2016'] = results[(results['Date'] >= dt(2016, 2, 8)) & (results['Date'] <= dt(2017, 11, 21))]
resultsDict['2017'] = results[(results['Date'] >= dt(2017, 9, 13)) & (results['Date'] <= dt(2017, 12, 7))]

for k,v in resultsDict.items():
    print(f"{k:{10}}{len(v)} speeches")
    resultsDict[k].to_csv(f"results/{k}/ssm_results_{k}.csv", index=False)


# %% ----- Construct Graphs for different timeframes
importlib.reload(constructGraph)

G_dict = {}
centDict = {}
cliquesDict = {}

for tf in TIME_FRAMES:
    print(f"========== Constructing graph for {tf} ==========")
    results = pd.read_csv(f"{PATH}{tf}/ssm_results_{tf}.csv")
    G_dict[tf], centDict[tf], cliquesDict[tf] = constructGraph.constructG(results, sentiDiff_thres)

    # Save graph
    nx.write_gpickle(G_dict[tf], f"{PATH}{tf}/ssm_weightedGraph_{tf}.gpickle")
    # Save centrality results
    centDict[tf].to_csv(f"{PATH}{tf}/ssm_centrality_{tf}.csv")
    # Save clique results
    with open(f"{PATH}{tf}/ssm_cliques_{tf}.pickle", "wb") as file:
        pickle.dump(cliquesDict[tf], file)


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

#%%
data = {'coeffMax' : 0, 'coeffDiff': 1}
coeffPrps_thres = pd.Series(data)
coeffPrps_thres['coeffMax']