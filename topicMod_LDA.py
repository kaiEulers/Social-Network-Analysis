"""
LDA doesn't produce very good results!!!
@author: kaisoon
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def topicMod(DATA, nTOPICS, nWORDS, COEFFMAX_PERC_THRES, COEFFDIFF_PERC_THRES):
    # ================================================================================
    # ----- FOR DEBUGGING
    # PATH = f"results/"

    # PARAMETERS
    # DATA = pd.read_csv(f"data/ssm_rel.csv")
    # nTOPICS = 5
    # nWORDS = 15
    # COEFFMAX_PERC_THRES = 0.1
    # COEFFDIFF_PERC_THRES = 0.1
    # ================================================================================
    # ----- Topic Modelling using Latent Dirichlet Allocation(LDA)
    # Instantiate count vectorisator model
    cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    # Instantiate LDA model
    LDA = LatentDirichletAllocation(n_components=nTOPICS)
    # Create document term matrix with count vectorisor model
    dtm = cv.fit_transform(DATA['Speech'])

    # Apply Latent Dirichlet Allocation on the document term matrix
    LDA.fit(dtm)
    # LDA.transform() returns a matrix that indicates the probability of each document belonging to a topic
    topicResults = LDA.transform(dtm)

    # ================================================================================
    # To remove ambiguous speeches, the top coefficients are obtained with a certain percentile of this removed. Then the difference between the top-two coefficients are obtained and again, the certain percentile is removed.
    # Sort each row from topResults
    coeff_sorted = [sorted(row) for row in topicResults]
    # Construct dataFrame with highest coeff and the difference between the top-two coeff
    coeff_stats = {
        'coeffMax' : [row[-1] for row in coeff_sorted],
        'coeffDiff': [row[-1] - row[-2] for row in coeff_sorted],
    }
    coeff_stats = pd.DataFrame(coeff_stats, columns='coeffMax coeffDiff'.split())

    # Sort the dataframe by the highest coeff in descending order
    coeff_stats.sort_values(by='coeffMax', inplace=True)
    # Extract coeffMax to be saved later
    coeffMax = pd.DataFrame(coeff_stats['coeffMax'], columns='coeffMax'.split())
    # We want the coeffMax to be high, therefore only take data above the threshold
    # TODO: Need to change the way the threshold is computed here?
    coeffMax_thres = round(len(coeff_stats)*COEFFMAX_PERC_THRES)
    coeff_stats = coeff_stats[coeffMax_thres:]

    # Sort the dataframe by the difference between the top-two coeff in descending order and only take data above the threshold
    coeff_stats.sort_values(by='coeffDiff', inplace=True)
    # Extract coeffDiff to be saved later
    coeffDiff = pd.DataFrame(coeff_stats['coeffDiff'], columns='coeffDiff'.split())
    # We want the coeffDiff to be high, therefore only take data above the threshold
    # TODO: Need to change the way the threshold is computed here?
    coeffDiff_thres = round(len(coeff_stats)*COEFFDIFF_PERC_THRES)
    coeff_stats = coeff_stats[coeffDiff_thres:]

    # Retrieve corresponding data from results
    results = DATA.iloc[coeff_stats.index]
    # Assign topic/s to each speech
    topicAssigned = [topicResults[i].argmax() for i, row in coeff_stats.iterrows()]

    # Concat assigned topics
    speeches = results['Speech']
    results = results.drop('Speech', axis=1)
    results["Topic"] = topicAssigned
    results["CoeffMax"] = coeffMax
    results["CoeffDiff"] = coeffDiff
    results["Speech"] = speeches
    results.sort_index(inplace=True)

    # ================================================================================
    # ----- Analyse results from NMF topic modelling
    # Store top w words in dataFrame topics_nmf
    # Number of words that describes topic
    topics = pd.DataFrame()
    for index, topic in enumerate(LDA.components_):
        # Negating an array causes the highest value to be the lowest value and vice versa
        topWordsIndex = (-topic).argsort()[:nWORDS]
        topics = topics.append(
            pd.Series([cv.get_feature_names()[i] for i in topWordsIndex]), ignore_index=True)
    topics = topics.transpose()

    # Compute topicCount and percentage of each topic
    topicCount = {t: topicAssigned.count(t) for t in range(nTOPICS)}
    percentage = {t: round(topicCount[t]/sum(topicCount.values()), 2) for t in range(nTOPICS)}

    topics.loc['topicCount'] = topicCount
    topics.loc['percentage'] = percentage

    print(topics)

    # ================================================================================
    # ----- FOR DEBUGGING
    # Save results
    # results.to_csv(f"{PATH}ssm_results_lda.csv")
    # topics.to_csv(f"{PATH}ssm_topics_lda.csv")
    # coeffMax.to_csv(f"{PATH}/distributions/ssm_topicCoeffMax_lda.csv")
    # coeffDiff.to_csv(f"{PATH}/distributions/ssm_topicCoeffDiff_lda.csv")
    # ================================================================================

    return results, topics, coeffMax, coeffDiff
