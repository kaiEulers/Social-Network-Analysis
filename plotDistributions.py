import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import colourPals as cp
import topicMod
import sentiAnalysis

sns.set_style("darkgrid")
sns.set_context("notebook")
FIG_SIZE = 3
LABEL_SIZE = 8
PATH = f"results/stats/"

#%% Distribution plots of coefMax, coeffDiff, sentiDiff, and centrality
importlib.reload(cp)
# Select between 'nmf' or 'lda' topic modelling
METHOD = 'nmf'

coeffMax = pd.read_csv(f"{PATH}ssm_topicCoeffMax_{METHOD}.csv", index_col=0)
coeffDiff = pd.read_csv(f"{PATH}ssm_topicCoeffDiff_{METHOD}.csv", index_col=0)
sentiDiff = pd.read_csv(f"{PATH}ssm_sentiDiff_{METHOD}.csv", index_col=0)

coeffMax.describe()
coeffDiff.describe()
sentiDiff.describe()

fig = plt.figure(figsize=(FIG_SIZE*3, FIG_SIZE*2), dpi=300, tight_layout=True)

# ----- Distribution of coeffMax
plt.subplot(2, 2, 1)
ax = sns.distplot(coeffMax, kde=False, color=cp.cbPaired['blue'])
# ax.set_ylim(0, 15)
ax.set_xlabel(f"Highest Topic Coefficient {METHOD.upper()}", fontsize=LABEL_SIZE)
ax.set_ylabel("Frequency", fontsize=LABEL_SIZE)

# ----- Distribution of coeffDiff
plt.subplot(2, 2, 2)
ax = sns.distplot(coeffDiff, kde=False, color=cp.cbPaired['purple'])
# ax.set_ylim(0, 15)
ax.set_xlabel(f"Coefficient Difference of Top-two Topics {METHOD.upper()}", fontsize=LABEL_SIZE)
ax.set_ylabel(None)

# ----- Distribution of sentiDiff
plt.subplot(2, 2, (3,4))
ax = sns.distplot(sentiDiff, kde=False, color=cp.cbPaired['green'])
# ax.set_ylim(0, 15)
ax.set_xlabel(f"Sentiment Intensity Difference {METHOD.upper()}", fontsize=LABEL_SIZE)
ax.set_ylabel("Frequency", fontsize=LABEL_SIZE)

fig.savefig(f"{PATH}ssm_distributions_{METHOD}.png", dpi=300)
fig.show()


# %% ----- Topic Modeling with varying number of topics
importlib.reload(topicMod)
importlib.reload(sentiAnalysis)
data = pd.read_csv(f"data/ssm_rel.csv")

# Parameters
nTOPICS = [5, 10, 15, 20, 25, 30]
nWORDS = 15
COEFFMAX_PERC_THRES = 0.1
COEFFDIFF_PERC_THRES = 0.1
SENTIDIFF_PERC_THRES = 0.1
METHOD = 'nmf'

# ----- Compute coeffMax, coeffDiff, sentiDiff for varying nTOPICS
coeffMax = {}
coeffDiff = {}
sentiDiff = {}
topics = {}
results_topicMod_senti = {}
sentiDiff_thres = {}
for n in nTOPICS:
    results_topicMod, topics[n], coeffMax[n], coeffDiff[n] = topicMod.model(data, n, nWORDS, COEFFMAX_PERC_THRES, COEFFDIFF_PERC_THRES, method=METHOD)
    results_topicMod_senti[n], sentiDiff_thres[n], sentiDiff[n] = sentiAnalysis.sentiAnal(
        results_topicMod, SENTIDIFF_PERC_THRES)

    # Save topics statistics
    topics[n].to_csv(f"{PATH}ssm_topics_{METHOD}_{n}.csv")

# ----- Plot Distributions
fig = plt.figure(figsize=(FIG_SIZE*4.5, FIG_SIZE*len(nTOPICS)), dpi=300, tight_layout=True)
subplot_rows = len(nTOPICS)
subplot_cols = 3
subgraph = 0
for n in nTOPICS:
    # ----- Distribution of coeffMax
    subgraph += 1
    plt.subplot(subplot_rows, subplot_cols, subgraph)
    ax = sns.distplot(coeffMax[n], kde=False, color=cp.cbPaired['blue'])
    ax.set_title(f"{n} Topics Modelled")
    ax.set_xlabel(f"Highest Topic Coefficient {METHOD.upper()}", fontsize=LABEL_SIZE)
    ax.set_ylabel("Frequency", fontsize=LABEL_SIZE)
    # ax.set_xlim(0, 0.75)
    # ax.set_ylim(0, 70)

    # ----- Distribution of coeffDiff
    subgraph += 1
    plt.subplot(subplot_rows, subplot_cols, subgraph)
    ax = sns.distplot(coeffDiff[n], kde=False, color=cp.cbPaired['purple'])
    ax.set_title(f"{n} Topics Modelled")
    ax.set_xlabel(f"Difference in Top-two Topic Coefficients {METHOD.upper()}",
                  fontsize=LABEL_SIZE)
    # ax.set_xlim(0, 0.75)
    # ax.set_ylim(0, 85)

    # ----- Distribution of sentiDiff
    subgraph += 1
    plt.subplot(subplot_rows, subplot_cols, subgraph)
    ax = sns.distplot(sentiDiff[n], kde=False, color=cp.cbPaired['green'])
    ax.set_title(f"{n} Topics Modelled")
    ax.set_xlabel(f"Difference in Sentiment {METHOD.upper()}", fontsize=LABEL_SIZE)
    # ax.set_xlim(0, 0.65)
    # ax.set_ylim(0, 9500)

plt.show()
fig.savefig(f"{PATH}ssm_distributions_varyingNumOfTopics_{METHOD}")
os.system('say "Analysis complete"')


#%%
