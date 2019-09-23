"""
@author: kaisoon
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import PowerTransformer
import time as tm
import importlib
import colourPals as cp
importlib.reload(cp)


def nmf(DATA, nTOPICS, nWORDS):
    # ----- Topic Modelling using Non-negative Matrix Factorisation(NMF)
    # Instantiate Tfidf model
    tfidf = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
    # Create document term matrix with tfidf model
    dtm = tfidf.fit_transform(DATA)

    # Instantiate NMF model
    nmf_model = NMF(n_components=nTOPICS)
    # Apply non-negative matrix factorisation on the document term matrix
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
    # Create document term matrix with count vectorisor model
    dtm = cv.fit_transform(DATA)

    # Instantiate LDA model
    lda_model = LatentDirichletAllocation(n_components=nTOPICS)
    # Apply Latent Dirichlet Allocation on the document term matrix
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
    :param TRANSFORM_METHOD: choose between 'yeo-johnson' or 'box-cox'. 'box-cox' only works for positive data.
    :param PLOT: option to visualise distribution transformation and threshold to remove outliers
    :return: results is a dataframe containing the topic numbers and its corresponding speech
    :return: topics is a dataframe containing the top-nWORDS words that describes each topic, the number of data that is of each topic, and the percentage of the entire dataset.
    :return: coeffPrps is a dataframe of top coefficient and the difference between the top-2 coefficient of each data
    """
    # ================================================================================
    # ----- FOR DEBUGGING
    # PATH = f"results/"

    # PARAMETERS
    # DATA = pd.read_csv(f"data/ssm_rel_lemma.csv")
    # nTOPICS = 4
    # nWORDS = 10
    # METHOD = 'nmf'
    # PLOT = True
    # COEFFMAX_STD = -1.5
    # COEFFDIFF_STD = -1
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
    # Construct dataFrame with highest coeff and the difference between the top-two coeff
    coeffPrps = {
        'coeffMax': [row[-1] for row in coeff_sorted],
        'coeffDiff': [row[-1] - row[-2] for row in coeff_sorted],
    }
    coeffPrps = pd.DataFrame(coeffPrps, columns='coeffMax coeffDiff'.split())

    # ----- Transformation distribution of coeffMax & coeffDiff to normal
    pt = PowerTransformer(method=TRANSFORM_METHOD)

    # Find optimal lambda value of Yeo-Johnson transform
    pt.fit(coeffPrps)
    pt_lambda = pt.lambdas_
    # Tranform each column to a normal distribution
    coeffPrps_trans = pt.transform(coeffPrps)
    # Concat transformed data
    coeffPrps['coeffMax_trans'] = coeffPrps_trans[:, 0]
    coeffPrps['coeffDiff_trans'] = coeffPrps_trans[:, 1]

    # ----- Remove outliers
    # Find index of outliers 2 std away in the neg direction
    coeffPrps_stats = coeffPrps.describe()

    # Compute coeffMax threshold and index speeches to be removed
    coeffMax_thres_trans = coeffPrps_stats['coeffMax_trans'].loc['mean'] + COEFFMAX_STD*coeffPrps_stats['coeffMax_trans'].loc['std']
    coeffMax_remove_i = set(coeffPrps[coeffPrps['coeffMax_trans'] < coeffMax_thres_trans].index)

    # Compute coeffDiff threshold and index speeches to be removed
    coeffDiff_thres_trans = coeffPrps_stats['coeffDiff_trans'].loc['mean'] + COEFFDIFF_STD*coeffPrps_stats['coeffDiff_trans'].loc['std']
    coeffDiff_remove_i = set(coeffPrps[coeffPrps['coeffDiff_trans'] < coeffDiff_thres_trans].index)

    # Take the union of speech indices of both coeffMax and coeffDiff
    remove_i = list(coeffMax_remove_i.union(coeffDiff_remove_i))
    remove_i.sort()
    # Drop data that with these indicies
    results = DATA.drop(remove_i)

    # Inverse transform the thresholds to get the untransformed threshold
    coeffPrps_thres = pt.inverse_transform(np.array([coeffMax_thres_trans, coeffDiff_thres_trans]).reshape(1, -1))
    coeffPrps_thres = pd.Series({
        'coeffMax' : coeffPrps_thres[0, 0],
        'coeffDiff': coeffPrps_thres[0, 1]
    })
    # TODO: run this again to check that it works

    # ----- Optional visualisation of distribution transformation and threshold to remove outliers
    if PLOT:
        FIG_SIZE = 3
        LABEL_SIZE = 8
        sns.set_style("darkgrid")
        sns.set_context("notebook")
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(FIG_SIZE*3, FIG_SIZE*3), dpi=300)

        col = 0
        sns.distplot(coeffPrps['coeffMax'], rug=True, ax=ax[0, col], color=cp.cbPaired['blue'])
        ax[0, col].axvline(x=coeffPrps_thres['coeffMax'], c=cp.cbPaired['red'])
        ax[0, col].set_title(f"{nTOPICS} Topics Modelled with {METHOD.upper()}\nRemoval of Outliers below Threshold")
        ax[0, col].set_xlabel(f"Highest Coefficient", fontsize=LABEL_SIZE)
        ax[0, col].set_ylabel(f"Kernel Density", fontsize=LABEL_SIZE)

        sns.distplot(coeffPrps['coeffMax_trans'], rug=True, ax=ax[1, col],color=cp.cbPaired['purple'])
        ax[1, col].axvline(x=coeffMax_thres_trans, c=cp.cbPaired['red'])
        ax[1, col].set_title(f"lambda = {round(pt_lambda[col], 3)}")
        ax[1, col].set_xlabel(f"Highest Coefficient Transformed", fontsize=LABEL_SIZE)
        ax[1, col].set_ylabel(f"Kernel Density", fontsize=LABEL_SIZE)

        qqplot(coeffPrps['coeffMax_trans'], ax=ax[2, col], line='s', color=cp.cbPaired['purple'])

        col = 1
        sns.distplot(coeffPrps['coeffDiff'], rug=True, ax=ax[0, col], color=cp.cbPaired['blue'])
        ax[0, col].axvline(x=coeffPrps_thres['coeffDiff'], c=cp.cbPaired['red'])
        ax[0, col].set_title(f"{nTOPICS} Topics Modelled with {METHOD.upper()}\nRemoval of Outliers below Threshold")
        ax[0, col].set_xlabel(f"Difference of Top-two Coefficient", fontsize=LABEL_SIZE)
        ax[0, col].set_ylabel(f"Kernel Density", fontsize=LABEL_SIZE)

        sns.distplot(coeffPrps['coeffDiff_trans'], rug=True, ax=ax[1, col],color=cp.cbPaired['purple'])
        ax[1, col].axvline(x=coeffDiff_thres_trans, c=cp.cbPaired['red'])
        ax[1, col].set_title(f"lambda = {round(pt_lambda[col], 3)}")
        ax[1, col].set_xlabel(f"Highest Coefficient Transformed", fontsize=LABEL_SIZE)
        ax[1, col].set_ylabel(f"Kernel Density", fontsize=LABEL_SIZE)

        qqplot(coeffPrps['coeffDiff_trans'], ax=ax[2, col], line='s', color=cp.cbPaired['purple'])

        plt.tight_layout()
        plt.show()
        fig.savefig(f"results/ssm_{nTOPICS}topics_{METHOD}_coeff_analysis.png")

    # ----- Assign topic/s to each speech
    topicAssigned = [topicResults[i].argmax() for i,row in results.iterrows()]

    # Concat assigned topics
    speeches = results[['Speech', 'Speech_lemma']]
    results = results.drop(['Speech', 'Speech_lemma'], axis=1)
    results["Topic"] = topicAssigned
    results["CoeffMax"] = coeffPrps['coeffMax']
    results["CoeffDiff"] = coeffPrps['coeffDiff']
    results = pd.concat([results, speeches], axis=1)
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
            highestCoeffMax_row = oneTopic.loc[highestCoeffMax_i]

            coeffMax_highest[t] = round(highestCoeffMax_row['CoeffMax'], 3)
            coeffDiff_highest[t] = round(highestCoeffMax_row['CoeffDiff'], 3)
            speechId_highest[t] = round(highestCoeffMax_row['Speech_id'], 3)

    # Concat statistics to topics
    topics.loc['coeffMax'] = coeffMax_highest
    topics.loc['coeffDiff'] = coeffDiff_highest
    topics.loc['speechId'] = speechId_highest
    topics.loc['topicCount'] = topicCount
    topics.loc['percentage'] = percentage

    print(topics)
    print(f"{round(len(remove_i)/len(results)*100, 1)}% of speeches removed")
    dur = tm.gmtime(tm.perf_counter() - startTime)
    print(f"\nTopic Modelling complete!\nModelling took {dur.tm_sec}s")

    # ================================================================================
    # ----- FOR DEBUGGING
    # Save results
    # results.to_csv(f"{PATH}ssm_results_{METHOD}.csv", index=False)
    # topics.to_csv(f"{PATH}ssm_{nTOPICS}topics_{METHOD}.csv")
    # coeffPrps.to_csv(f"{PATH}/stats/ssm_topicCoeffPrps_{METHOD}.csv")
    # ================================================================================
    return results, topics, coeffPrps_thres, coeffPrps
