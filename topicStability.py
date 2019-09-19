import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import importlib
import time
import os
import jaccard_similarity as js
import topicMod
import colourPals as cp
importlib.reload(js)
importlib.reload(topicMod)
importlib.reload(cp)

def crossValid_split(data, k=10):
    """
    :param data:
    :param k: number of fold cross validation
    :return: a generator that contains the data split for k-fold cross validation
    """
    lastIndex = len(data) - 1
    partLength = round(len(data)/k)

    # Construct start and end index for each part to drop from original data
    indexList = [0]
    while lastIndex > 0:
        indexList.append(lastIndex)
        lastIndex -= partLength
    indexList.sort()

    # Yield a generator containing k subsets of data each with a different part dropped from the original data
    for i in range(len(indexList) - 1):
        toDrop = list(range(indexList[i], indexList[i+1]))
        yield data.drop(toDrop)


# ----- Program starts here
startTime = time.time()
PATH = "results/stats/"
# Select between 'nmf' or 'lda' topic modelling
METHOD = "nmf"

data_complete = pd.read_csv("data/ssm_rel.csv")
data_complete = data_complete.reset_index(drop=True)
data = data_complete['Speech']
nTOPICS_RANGE = {'start': 2, 'end': 25, 'step': 1}
nWORDS = 20
K_FOLD = 10

stabilityDict = {}
for n in range(nTOPICS_RANGE['start'], nTOPICS_RANGE['end'] + 1, nTOPICS_RANGE['step']):
    print(f"===== Computing stability for {n} topics =====")
    # ----- Topic modelling for s0
    if METHOD == 'lda':
        _, topics_data = topicMod.lda(data, n, nWORDS)
    else:
        _, topics_data = topicMod.nmf(data, n, nWORDS)
    # Save topics_data
    pd.DataFrame.from_dict(topics_data).to_csv(f"{PATH}topics_{METHOD}/ssm_{n}topics_{METHOD}.csv")

    # ----- Topic modelling for s1, s2, s3, ..., sn
    agreeScore_sum = 0
    # crossValid_split() returns a generator that contains the data split for k-fold cross validation
    for j, subData in enumerate(crossValid_split(data, k=K_FOLD)):
        print(f"Processing subData {j}...")

        if METHOD == 'lda':
            _, topics_subData = topicMod.lda(subData, n, nWORDS)
            # Sum all agreement scores
            agreeScore_sum += js.agree(topics_data, topics_subData)
        else:
            _, topics_subData = topicMod.nmf(subData, n, nWORDS)
            # Sum all agreement scores
            agreeScore_sum += js.agree(topics_data, topics_subData)

    # Compute stability as the average of all agreement scores
    stabilityDict[n] = agreeScore_sum/K_FOLD
    print(f"===== Stability for {n} topics computed! =====\n")

# Save stability results
stability = pd.DataFrame.from_dict(stabilityDict, orient='index', columns='stability'.split())
stability.to_csv(f"{PATH}ssm_topicStability_top{nWORDS}words.csv")

endTime = time.time() - startTime
print(f"Optimal number of topics to model is {int(stability.idxmax())}!")
print(f"Stability analysis completed in {int(endTime//60)}mins {round(endTime%60)}s for {nTOPICS_RANGE} topics")
os.system("say 'Stability analysis complete'")


#%% ----- Plot stability values
sns.set_style("darkgrid")
sns.set_context("notebook")
FIG_SIZE = 3
LABEL_SIZE = 8

stability = pd.read_csv(f"{PATH}ssm_topicStability_top{nWORDS}words.csv")
fig = plt.figure(figsize=(FIG_SIZE*3, FIG_SIZE*2), dpi=300, tight_layout=True)

ax = sns.barplot(stability.index, stability['stability'], color=cp.cbPaired['orange'])
ax.set_title(f"Topic Stability Analysis using Top-{nWORDS} Words\nSame-sex Marriage Bills")
ax.set_ylabel('Stability', fontsize=LABEL_SIZE)
ax.set_xlabel('Number of Topics Modelled', fontsize=LABEL_SIZE)

fig.savefig(f"{PATH}ssm_topicStability_top{nWORDS}words.png", dpi=300)
fig.show()

