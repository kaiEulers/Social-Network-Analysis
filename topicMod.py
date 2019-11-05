"""
@author: kaisoon
"""
import importlib
import time as tm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from statsmodels.graphics.gofplots import qqplot

import colourPals as cp
import kaiFunctions as mf

importlib.reload(mf)
importlib.reload(cp)


def nmf(DATA, nTOPICS, nWORDS):
    # ----- Topic Modelling using Non-negative Matrix Factorisation(NMF)
    # Instantiate Tfidf model
    tfidf = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
    # Create document timeframes matrix with tfidf model
    dtm = tfidf.fit_transform(DATA)

    # Instantiate NMF model
    nmf_model = NMF(n_components=nTOPICS)
    # Apply non-negative matrix factorisation on the document timeframes matrix
    nmf_model.fit(dtm)
    # nmf_model.transform() returns a matrix with coefficients that shows how much each document belongs to a topic
    topicResults = nmf_model.transform(dtm)

    # Store top nWords in dataFrame topics. These are the words thata describes the topic.
    topics = {}
    for i, t in enumerate(nmf_model.components_):
        # Negating an array causes the highest value to be the lowest value and vice versa
        topWordsIndex = (-t).argsort()[:nWORDS]
        topics[i] = [tfidf.get_feature_names()[i] for i in topWordsIndex]

    return topicResults, topics


def lda(DATA, nTOPICS, nWORDS):
    # ----- Topic Modelling using Latent Dirichlet Allocation(LDA)
    # Instantiate count vectorisator model
    cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    # Create document timeframes matrix with count vectorisor model
    dtm = cv.fit_transform(DATA)

    # Instantiate LDA model
    lda_model = LatentDirichletAllocation(n_components=nTOPICS)
    # Apply Latent Dirichlet Allocation on the document timeframes matrix
    lda_model.fit(dtm)
    # lda_model.transform() returns a matrix that indicates the probability of each document belonging to a topic
    topicResults = lda_model.transform(dtm)

    # Store top nWords in dataFrame topics. These are the words thata describes the topic.
    topics = {}
    for i, t in enumerate(lda_model.components_):
        # Negating an array causes the highest value to be the lowest value and vice versa
        topWordsIndex = (-t).argsort()[:nWORDS]
        topics[i] = [cv.get_feature_names()[i] for i in topWordsIndex]

    return topicResults, topics


def model(DATA, nTOPICS, nWORDS, COEFFMAX_STD, COEFFDIFF_STD, METHOD='nmf', TRANSFORM_METHOD='yeo-johnson', PLOT=False):
    """
    :param DATA: is a dataframe with columns 'Person' and 'Speech'
    :param nTOPICS: is the # of topics that will be modelled
    :param nWORDS: is the top nWORDS words used to describe each topic that is modelled
    :param COEFFMAX_STD: top topic coefficient that is below this normalised standard deviation will be removed
    :param COEFFDIFF_STD: the difference between the top-2 topic coefficient that is below this normalised standard deviation will be removed
    :param METHOD: choose between 'nmf' or 'lda'
    :param TRANSFORM_METHOD: choose between 'yeo-johnson' or 'box-cox'. 'box-cox' only works for positive text.
    :param PLOT: option to visualise distribution transformation and threshold to remove outliers
    :return: results is a dataframe containing the topic numbers and its corresponding speech
    :return: topics is a dataframe containing the top-nWORDS words that describes each topic, the number of text that is of each topic, and the percentage of the entire dataset.
    :return: coeffPrps is a dataframe of top coefficient and the difference between the top-2 coefficient of each text
    """
    # ================================================================================
    # ----- FOR DEBUGGING
    PATH = f"results/"

    # PARAMETERS
    METHOD = 'nmf'
    # nTOPICS = 11
    # nWORDS = 10
    # COEFFMAX_STD = -1.5
    # COEFFDIFF_STD = -1.5
    # PLOT = True
    # text = pd.read_csv(f"text/ssm_byPara_rel_lemma.csv")
    # ================================================================================
    # ----- Error Checking
    assert COEFFMAX_STD < 0, f"Normalised standard deviation threshold of coeffMax has to be negative. {COEFFMAX_STD} as passed."
    assert COEFFDIFF_STD < 0, f"Normalised standard deviation threshold of coeffDiff has to be negative. {COEFFDIFF_STD} as passed."
    # ================================================================================
    # Perform topic modelling base on selected method
    startTime = tm.perf_counter()
    if METHOD == 'nmf':
        topicResults, topics = nmf(DATA['Speech_lemma'], nTOPICS, nWORDS)
        topics = pd.DataFrame(topics)

    elif METHOD == 'lda':
        topicResults, topics = lda(DATA['Speech_lemma'], nTOPICS, nWORDS)
        topics = pd.DataFrame(topics)

    else:
        raise ValueError("Method parameter must be either 'nmf' or 'lda'!")

    # ================================================================================
    # ----- To remove ambiguous speeches, the top coefficients are obtained with a certain percentile of this removed. Then the difference between the top-two coefficients are obtained and again, the certain percentile is removed.
    # Sort each row from topResults
    coeff_sorted = [sorted(row) for row in topicResults]
    # Construct dataFrame with highest coeff
    coeffMax = pd.Series([row[-1] for row in coeff_sorted])
    # Remove outliers
    coeffMax_noOutliers, coeffMax_thres, coeffMax_trans = mf.get_outliers(coeffMax, COEFFMAX_STD, METHOD=TRANSFORM_METHOD)
    coeffMax_noOutliers = coeffMax_noOutliers.rename('coeffMax')

    # Construct daataFrame with the difference between the top-two coeff
    coeffDiff = pd.Series([row[-1] - row[-2] for row in coeff_sorted])
    # Remove outliers
    coeffDiff_noOutliers, coeffDiff_thres, coeffDiff_trans = mf.get_outliers(coeffDiff, COEFFDIFF_STD, METHOD=TRANSFORM_METHOD)
    coeffDiff_noOutliers = coeffDiff_noOutliers.rename('coeffDiff')

    # Merge coeffMax and coeffDiff by intersection
    coeffPrps = pd.merge(coeffMax_noOutliers, coeffDiff_noOutliers, how='inner', left_index=True, right_index=True)
    coeffPrps_thres = pd.Series([coeffMax_thres, coeffDiff_thres], index='coeffMax coeffDiff'.split())
    # Remove all outliers from text
    results = DATA.loc[coeffPrps.index]

    # ----- Optional visualisation of distribution transformation and threshold to remove outliers
    if PLOT:
        FIG_SIZE = 3
        LABEL_SIZE = 8
        sns.set_style("darkgrid")
        sns.set_context("notebook")
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(FIG_SIZE*3, FIG_SIZE*3), dpi=300)

        # Plot coeffMax before transformation
        col = 0
        sns.distplot(coeffMax, rug=True, ax=ax[0, col], color=cp.cbPaired['blue'])
        ax[0, col].axvline(x=coeffMax_thres, c=cp.cbPaired['red'])
        ax[0, col].set_title(f"{nTOPICS} Topics Modelled with {METHOD.upper()}\nOutliers removed below {COEFFMAX_STD}std")
        ax[0, col].set_xlabel(f"Highest Coefficient", fontsize=LABEL_SIZE)
        ax[0, col].set_ylabel(f"Kernel Density", fontsize=LABEL_SIZE)
        # Plot coeffMax after transformation
        sns.distplot(coeffMax_trans, rug=True, ax=ax[1, col] ,color=cp.cbPaired['purple'])
        ax[1, col].axvline(x=COEFFMAX_STD, c=cp.cbPaired['red'])
        ax[1, col].set_xlabel(f"Highest Coefficient Transformed", fontsize=LABEL_SIZE)
        ax[1, col].set_ylabel(f"Kernel Density", fontsize=LABEL_SIZE)
        # Plot qqplot of coeffMax after transformation
        qqplot(coeffMax_trans, ax=ax[2, col], line='s', color=cp.cbPaired['purple'])


        # Plot coeffDiff before transformation
        col = 1
        sns.distplot(coeffDiff, rug=True, ax=ax[0, col], color=cp.cbPaired['blue'])
        ax[0, col].axvline(x=coeffDiff_thres, c=cp.cbPaired['red'])
        ax[0, col].set_title(f"{nTOPICS} Topics Modelled with {METHOD.upper()}\nOutliers removed below {COEFFDIFF_STD}std")
        ax[0, col].set_xlabel(f"Difference of Top-two Coefficient", fontsize=LABEL_SIZE)
        ax[0, col].set_ylabel(f"Kernel Density", fontsize=LABEL_SIZE)
        # Plot coeffDiff after transformation
        sns.distplot(coeffDiff_trans, rug=True, ax=ax[1, col] ,color=cp.cbPaired['purple'])
        ax[1, col].axvline(x=COEFFDIFF_STD, c=cp.cbPaired['red'])
        ax[1, col].set_xlabel(f"Highest Coefficient Transformed", fontsize=LABEL_SIZE)
        ax[1, col].set_ylabel(f"Kernel Density", fontsize=LABEL_SIZE)
        # Plot qqplot of coeffDiff after transformation
        qqplot(coeffDiff_trans, ax=ax[2, col], line='s', color=cp.cbPaired['purple'])

        plt.tight_layout()
        plt.show()

    # ----- Assign topic/s to each speech
    topicAssigned = [topicResults[i].argmax() for i ,row in results.iterrows()]

    # Concat assigned topics
    results = DATA
    results.insert(10, "Topic", topicAssigned)
    results.insert(11, "CoeffMax", coeffMax)
    results.insert(12, "CoeffDiff", coeffDiff)
    results.sort_index(inplace=True)

    # ================================================================================
    # ----- Analyse results from NMF topic modelling
    # Compute topicCount and percentage of each topic
    topicNum = list(range(nTOPICS))
    topicCount = {t: topicAssigned.count(t) for t in topicNum}
    percentage = {t: round(topicCount[t]/sum(topicCount.values())*100, 1) for t in topicNum}

    # Retrieve the highest coeffMax and its corresponding coeffDiff and speech_id for each topic
    coeffMax_highest = {}
    coeffDiff_highest = {}
    speechId_highest = {}
    paraId_highest = {}
    for t in topicNum:
        # Retrieve all speeches from a single topic
        oneTopic = results[results['Topic'] == t]
        # If there are no speeches from this topic, assign nan to stats. Otherwise compute highest stats.
        if oneTopic.empty:
            coeffMax_highest[t] = np.nan
            coeffDiff_highest[t] = np.nan
            speechId_highest[t] = np.nan
        else:
            # df.idxmax() returns the index of the maximum value
            highestCoeffMax_i = oneTopic['CoeffMax'].idxmax()
            # Retrieve row with the highest coeffMax
            highestCoeffMax_row = oneTopic.loc[highestCoeffMax_i]

            coeffMax_highest[t] = round(highestCoeffMax_row['CoeffMax'], 3)
            coeffDiff_highest[t] = round(highestCoeffMax_row['CoeffDiff'], 3)
            speechId_highest[t] = round(highestCoeffMax_row['Speech_id'], 3)
            if 'Para_id' in oneTopic.columns:
                paraId_highest[t] = round(highestCoeffMax_row['Para_id'], 3)

    # Concat statistics to topics
    topics.loc['topicCount'] = topicCount
    topics.loc['percentage'] = percentage
    topics.loc['coeffMax'] = coeffMax_highest
    topics.loc['coeffDiff'] = coeffDiff_highest
    topics.loc['speechId'] = speechId_highest
    if 'Para_id' in oneTopic.columns:
        topics.loc['paraId'] = paraId_highest

    print(topics)
    print(f"{round((1 - len(results)/len(DATA))*100, 2)}% of speeches removed")
    dur = tm.gmtime(tm.perf_counter() - startTime)
    print(f"\nTopic Modelling complete! Modelling took {dur.tm_sec}s")

    # ================================================================================
    # ----- FOR DEBUGGING
    # Save results
    # results.to_csv(f"{PATH}ssm_results_{METHOD}.csv", index=False)
    # topics.to_csv(f"{PATH}ssm_{nTOPICS}topics_{METHOD}.csv")
    # coeffPrps.to_csv(f"{PATH}stats/ssm_coeffPrps_{METHOD}.csv", header=False)
    # ================================================================================
    return results, topics, coeffPrps_thres, coeffPrps
