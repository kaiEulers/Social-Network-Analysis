import random as rand
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import importlib
import time as tm
import os
import concurrent.futures
import jaccard_similarity as js
import topicMod
import colourPals as cp
import miscFuncs as mf

#%% Parameters for Topic Modelling
importlib.reload(js)
importlib.reload(topicMod)
importlib.reload(cp)
importlib.reload(mf)

nTopicList = list(range(2, 25 + 1, 1))
nWORDS = 20
# Select between 'nmf' or 'lda' topic modelling
METHOD = "nmf"
VALIDATION = 'k-fold'

if VALIDATION == 'k-fold':
    # Parameter for k-fold cross validation of subData for Jaccard similarity comparision
    K_FOLD = 10
    K = 10
elif VALIDATION == 'randSel':
    # Parameters for random selection of subData for Jaccard similarity comparision
    BETA = 0.8
    K = 100

PATH = "results/stats/topic_stability/"
data_complete = pd.read_csv("data/ssm_rel_lemma.csv")
data_complete = data_complete.reset_index(drop=True)
# data = data_complete['Speech']
data = data_complete['Speech_lemma']


def top_stb_analysis(data, nTopicList, nWORDS, METHOD, VALIDATION, K, BETA=0.8):
    """
    :param data:
    :param nTopicList:
    :param nWORDS:
    :param METHOD: select from 'nmf' or 'lda'
    :param VALIDATION: select from 'k-fold' or 'randSel'
    :param K:
    :param BETA: BETA must be between 0-1 and is the proportion of data that is selected to compare with the original data to compute stability
    :return:
    """
    # ----- Error checks
    assert BETA > 0 and BETA < 1, f"Data proportion for randSel has to be between 0 and 1, not '{BETA}'."
    assert K > 0, f"k has to be a positive number, not '{K}'."
    # if BETA < 0 or BETA > 1:
    #     raise ValueError()
    # elif K <= 0:
    #     raise ValueError()

    stabilityDict = {}
    for n in nTopicList:
        print(f"===== Computing stability for {n} topics =====")
        # ----- Topic modelling for s0
        if METHOD == 'lda':
            _, topics_data = topicMod.lda(data, n, nWORDS)
        else:
            _, topics_data = topicMod.nmf(data, n, nWORDS)
        # Save topics_data
        pd.DataFrame.from_dict(topics_data).to_csv(f"{PATH}topics_{METHOD}/ssm_{n}topics_{METHOD}.csv")

        agreeScore_sum = 0
        if VALIDATION == 'randSel':
            # ----- Randomly select BETA*len(data)) for topic modelling for Jaccard similarity comparision with topics modelled using 100% of the data
            for j in range(K):
                # Print progress...
                if j%10 == 0:
                    print(f"Processing subData {j}...")

                subData_len = round(len(data.index)*BETA)
                selection = rand.sample(list(data.index), k=subData_len)
                subData = data.loc[selection]

                if METHOD == 'lda':
                    _, topics_subData = topicMod.lda(subData, n, nWORDS)
                    # Sum all agreement scores
                    agreeScore_sum += js.agree(topics_data, topics_subData)
                elif METHOD == 'nmf':
                    _, topics_subData = topicMod.nmf(subData, n, nWORDS)
                    # Sum all agreement scores
                    agreeScore_sum += js.agree(topics_data, topics_subData)
                else:
                    raise ValueError(f"METHOD cannot be '{METHOD}'. It has to be either 'nmf' or 'lda'!")

        elif VALIDATION == 'k-fold':
            # ---------- Topic modelling for s1, s2, s3, ..., sn
            # ----- Select 90% of data for topic modelling for Jaccard similarity comparision with topics modelled using 100% of the data
            # crossValid_split() returns a generator that contains the data split for k-fold cross validation
            for j, subData in enumerate(mf.crossValid_split(data, k=K_FOLD)):
                # Print progress...
                print(f"Processing subData {j}...")

                if METHOD == 'lda':
                    _, topics_subData = topicMod.lda(subData, n, nWORDS)
                    # Sum all agreement scores
                    agreeScore_sum += js.agree(topics_data, topics_subData)
                elif METHOD == 'nmf':
                    _, topics_subData = topicMod.nmf(subData, n, nWORDS)
                    # Sum all agreement scores
                    agreeScore_sum += js.agree(topics_data, topics_subData)
                else:
                    raise ValueError(f"METHOD cannot be '{METHOD}'. It has to be either 'nmf' or 'lda'!")

        else:
            raise ValueError(f"crossValid parameter cannot be '{VALIDATION}'. It must be either 'k-fold' or 'randSel'!")

        # Compute stability as the average of all agreement scores
        stabilityDict[n] = agreeScore_sum/K
        print(f"===== Stability for {n} topics computed! =====\n")

    return stabilityDict


# ----- Using multi-processing for faster topic stability analysis

startTime = tm.perf_counter()
stabilityDict = {}
with concurrent.futures.ProcessPoolExecutor() as executor:
    # Split nTopicList into 8 portions
    nTopicListList = mf.similarSum_split(nTopicList, os.cpu_count())
    # Submit to each nTopicList to an executor. executor.submit() returns a futures object containing the results of the executions.
    fList = [executor.submit(top_stb_analysis, data, nTL, nWORDS, METHOD, VALIDATION, K) for nTL in nTopicListList]

    # Retrieve all results from each futures object
    for f in concurrent.futures.as_completed(fList):
        stabilityDict.update(f.result())

# Save stability results
stability = pd.DataFrame.from_dict(stabilityDict, orient='index', columns='stability'.split())
stability = stability.sort_index()
stability.to_csv(f"{PATH}ssm_topicStability_top{nWORDS}words_{METHOD}_{VALIDATION}{K}.csv")
print(f"Optimal number of topics to model is {int(stability.idxmax())}!")

dur = tm.gmtime(tm.perf_counter() - startTime)
print(f"Stability analysis completed in {dur.tm_hour}hrs {dur.tm_min}mins {round(dur.tm_sec)}s")
os.system("say 'Stability analysis complete'")


#%% ----- Plot stability values
sns.set_style("darkgrid")
sns.set_context("notebook")
FIG_SIZE = 3
LABEL_SIZE = 8

fig = plt.figure(figsize=(FIG_SIZE*3, FIG_SIZE*2), dpi=300, tight_layout=True)
stability = pd.read_csv(f"{PATH}ssm_topicStability_top{nWORDS}words_{METHOD}_{VALIDATION}{K}.csv", index_col=0)

ax = sns.lineplot(stability.index, stability['stability'], color=cp.cbPaired['orange'])
ax = sns.scatterplot(stability.index, stability['stability'], color=cp.cbPaired['orange'])
ax.set_title(f"Topic Stability with Top-{nWORDS} Words using {METHOD.upper()} & {VALIDATION}{K}")
ax.set_ylabel('Stability', fontsize=LABEL_SIZE)
# ax.set_ylim(0)
ax.set_xlabel('Number of Topics Modelled', fontsize=LABEL_SIZE)
ax.set_xticks(nTopicList)

fig.savefig(f"{PATH}ssm_topicStability_top{nWORDS}words_{METHOD}_{VALIDATION}{K}.png", dpi=300)
fig.show()


#%% ----- Plot stability values with varying nWORDS
sns.set_style("darkgrid")
sns.set_context("notebook")
FIG_SIZE = 3
LABEL_SIZE = 8
nWordList = [5, 10, 15, 20]

fig = plt.figure(figsize=(FIG_SIZE*3, FIG_SIZE*2), dpi=300)
subgraph = 220
for nWORDS in nWordList:
    stability = pd.read_csv(f"{PATH}ssm_topicStability_top{nWORDS}words_{METHOD}_{VALIDATION}{K}.csv", index_col=0)

    subgraph += 1
    plt.subplot(subgraph)
    ax = sns.lineplot(stability.index, stability['stability'], color=cp.cbPaired['orange'])
    ax = sns.scatterplot(stability.index, stability['stability'], color=cp.cbPaired['orange'])
    ax.set_title(f"Topic Stability with Top-{nWORDS} Words using {METHOD.upper()} & {VALIDATION}{K}", fontsize=LABEL_SIZE + 1)
    ax.set_ylabel('Stability', fontsize=LABEL_SIZE)
    # ax.set_ylim((0.5, 0.9))
    ax.set_xlabel('Number of Topics Modelled', fontsize=LABEL_SIZE)
    ax.set_xticks(nTopicList)
    ax.set_xticklabels(ax.get_xticks(), fontsize=LABEL_SIZE-1)

fig.savefig(f"{PATH}ssm_topicStability_{METHOD}_{VALIDATION}{K}.png", dpi=300)
plt.tight_layout()
plt.show()

