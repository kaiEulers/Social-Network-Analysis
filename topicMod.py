"""
@author: kaisoon
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

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

def model(DATA, nTOPICS, nWORDS, COEFFMAX_PERC_THRES, COEFFDIFF_PERC_THRES, method='nmf'):
    """
    :param DATA: is a dataframe with columns 'Person' and 'Speech'
    :param nTOPICS: is the # of topics that will be modelled
    :param nWORDS: is the top nWORDS words used to describe each topic that is modelled
    :param COEFFMAX_PERC_THRES: top topic coefficient that is below this percentile threshold will be removed
    :param COEFFDIFF_PERC_THRES: the difference between the top-2 topic coefficient that is below this percentile threshold will be removed
    :return: results is a dataframe containing the topic numbers and its correspnding speech
    :return: topic_nmf is dataframe containing the top-nWORDS words that describes each topic, the number of data that is of each topic, and the percentage of the entire dataset.
    :return: coeffMax is a series of top coefficient of each data
    :return: coeffDiff is a series of the difference between the top-2 coefficient of each data
    """
    # ================================================================================
    # ----- FOR DEBUGGING
    # PATH = f"results/"

    # PARAMETERS
    # TODO: How do I determine how many topics to model?
    # TODO: How do I determine COEFFMAX_PERC_THRES? WE WANT ABSOLUTE COEFFICIENT THAT ARE HIGH!
    # TODO: How do I determine COEFFDIFF_PERC_THRES? WE WANT COEFF DIFFERENCE THAT ARE HIGH!
    # DATA = pd.read_csv(f"data/ssm_rel.csv")
    # nTOPICS = 30
    # nWORDS = 15
    # COEFFMAX_PERC_THRES = 0.1
    # COEFFDIFF_PERC_THRES = 0.1
    # method = 'lda'
    # ================================================================================
    # Perform topic modelling base on selected method
    if method == 'nmf':
        topicResults, topics = nmf(DATA['Speech'], nTOPICS, nWORDS)
        topics = pd.DataFrame(topics)

    elif method == 'lda':
        topicResults, topics = lda(DATA['Speech'], nTOPICS, nWORDS)
        topics = pd.DataFrame(topics)

    else:
        raise ValueError("Method parameter must be either 'nmf' or 'lda'!")

    # ================================================================================
    # To remove ambiguous speeches, the top coefficients are obtained with a certain percentile of this removed. Then the difference between the top-two coefficients are obtained and again, the certain percentile is removed.
    # Sort each row from topResults
    coeff_sorted = [sorted(row) for row in topicResults]
    # Construct dataFrame with highest coeff and the difference between the top-two coeff
    coeff_stats = {
        'coeffMax': [row[-1] for row in coeff_sorted],
        'coeffDiff': [row[-1] - row[-2] for row in coeff_sorted],
    }
    coeff_stats = pd.DataFrame(coeff_stats, columns='coeffMax coeffDiff'.split())

    # Sort the dataframe by the highest coeff in descending order
    coeff_stats.sort_values(by='coeffMax', inplace=True)
    # Extract coeffMax to be saved later
    coeffMax = pd.DataFrame(coeff_stats['coeffMax'], columns='coeffMax'.split())
    # Only take data above the threshold
    coeffMax_thres = round(len(coeff_stats)*COEFFMAX_PERC_THRES)
    coeff_stats = coeff_stats[coeffMax_thres:]

    # Sort the dataframe by the difference between the top-two coeff in descending order and only take data above the threshold
    coeff_stats.sort_values(by='coeffDiff', inplace=True)
    # Extract coeffDiff to be saved later
    coeffDiff = pd.DataFrame(coeff_stats['coeffDiff'], columns='coeffDiff'.split())
    # Only take data above the threshold
    coeffDiff_thres = round(len(coeff_stats)*COEFFDIFF_PERC_THRES)
    coeff_stats = coeff_stats[coeffDiff_thres:]

    # Retrieve corresponding data from results
    results = DATA.iloc[coeff_stats.index]
    # Assign topic/s to each speech
    topicAssigned = [topicResults[i].argmax() for i,row in coeff_stats.iterrows()]

    # Concat assigned topics
    speeches = results['Speech']
    results = results.drop('Speech', axis=1)
    results["Topic"] = topicAssigned
    results["CoeffMax"] = coeff_stats['coeffMax']
    results["CoeffDiff"] = coeff_stats['coeffDiff']
    results["Speech"] = speeches
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
    # TODO: BUG when using LDA with nTOPICS=30
    topics.loc['coeffMax'] = coeffMax_highest
    topics.loc['coeffDiff'] = coeffDiff_highest
    topics.loc['speechId'] = speechId_highest
    topics.loc['topicCount'] = topicCount
    topics.loc['percentage'] = percentage

    print(topics)

    # ================================================================================
    # ----- FOR DEBUGGING
    # Save results
    # results.to_csv(f"{PATH}ssm_results_nmf.csv")
    # topics.to_csv(f"{PATH}ssm_topics_nmf.csv")
    # coeffMax.to_csv(f"{PATH}/stats/ssm_topicCoeffMax_nmf.csv")
    # coeffDiff.to_csv(f"{PATH}/stats/ssm_topicCoeffDiff_nmf.csv")
    # ================================================================================

    return results, topics, coeffMax, coeffDiff
