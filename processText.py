"""
@author: kaisoon
"""
import importlib

import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import colourPals as cp
import kaiFunctions as kf

importlib.reload(kf)
importlib.reload(cp)
# ================================================================================
# ----- FOR DEBUGGING
# RES = 'para'
# METHOD = 'nmf'
# TRANSFORM_METHOD = 'box-cox'
# nTOPICS = 11
# nWORDS = 20
# COEFFMAX_STD = 0
# COEFFDIFF_STD = 0
# PLOT = True
# NUM = 3
# results = pd.read_csv(f"results/ssm_by{RES.capitalize()}_rel_lemma.csv")
# text = results['Speech_lemma']
# ================================================================================

def nmf(DATA, nTOPICS, nWORDS):
    # ----- Topic Modelling using Non-negative Matrix Factorisation(NMF)
    # Instantiate Tfidf model
    tfidf = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
    # tfidf = TfidfVectorizer(min_df=2, stop_words='english')
    # Create document timeframes matrix with tfidf model
    dtm = tfidf.fit_transform(DATA)

    # Instantiate NMF model
    nmf_model = NMF(n_components=nTOPICS)
    # Apply non-negative matrix factorisation on the document timeframes matrix
    nmf_model.fit(dtm)
    # nmf_model.transform() returns a matrix with coefficients that shows how much each document belongs to a topic
    topicMatrix = nmf_model.transform(dtm)

    # Store top nWords in dataFrame topics. These are the words thata describes the topic.
    topics = {}
    for i, t in enumerate(nmf_model.components_):
        # Negating an array causes the highest value to be the lowest value and vice versa
        topWordsIndex = (-t).argsort()[:nWORDS]
        topics[i] = [tfidf.get_feature_names()[i] for i in topWordsIndex]

    return topicMatrix, topics


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
    topicMatrix = lda_model.transform(dtm)

    # Store top nWords in dataFrame topics. These are the words thata describes the topic.
    topics = {}
    for i, t in enumerate(lda_model.components_):
        # Negating an array causes the highest value to be the lowest value and vice versa
        topWordsIndex = (-t).argsort()[:nWORDS]
        topics[i] = [cv.get_feature_names()[i] for i in topWordsIndex]

    return topicMatrix, topics

def topicModel(text, nTOPICS, nWORDS, METHOD='nmf'):
    # Perform topic modelling base on selected method
    if METHOD == 'nmf':
        topicMatrix, topics = nmf(text, nTOPICS, nWORDS)
        topics = pd.DataFrame(topics)
    elif METHOD == 'lda':
        topicMatrix, topics = lda(text, nTOPICS, nWORDS)
    else:
        raise ValueError("Method parameter must be either 'nmf' or 'lda'!")

    # ----- Assign topic/s to each speech base on highest coefficient
    topicAssigned = pd.Series([row.argmax() for row in topicMatrix])
    # ----- Statistics of topics assigned
    topics = pd.DataFrame(topics)
    topicCount = topicAssigned.value_counts().sort_index()
    topicPerc = topicCount/topicCount.sum()
    topics.loc['TopicCount'] = topicCount
    topics.loc['Percentage'] = round(topicPerc*100, 2)
    topics = topics.sort_values('TopicCount', axis=1, ascending=False)

    return topicMatrix, topics, topicAssigned


def sentiAnalysis(textSeries):
    # ----- Sentiment Analysis
    sid = SentimentIntensityAnalyzer()
    senti = pd.DataFrame(columns="pos neu neg compound".split())
    # ----- Analyse sentiment of all speeches
    for k, (i, text) in enumerate(textSeries.items()):
        score = sid.polarity_scores(text)
        senti.loc[i] = score
        # Print progress
        if k%50 == 0:
            print(f"{k:{5}} of {len(textSeries):{5}}\t{score}")
    return senti


def auditText(results, PATH, NUM=1, METHOD='nmf'):
    PATH = f"realityChecks/highestCoeffMax_{METHOD}/highestCoeffMax"
    topicList = results['Topic'].unique().tolist()
    topicList.sort()

    # Extract top NUM text with the highest coeff for auditing
    for topic in topicList:
        oneTopic_pos = results[(results['Topic'] == topic) & (results['Senti_comp'] > 0)]
        oneTopic_neg = results[(results['Topic'] == topic) & (results['Senti_comp'] < 0)]
        highest_pos = oneTopic_pos.nlargest(NUM, 'CoeffMax')
        highest_neg = oneTopic_neg.nlargest(NUM, 'CoeffMax')

        for i, ((_, row_pos), (_, row_neg)) in enumerate(
                zip(highest_pos.iterrows(), highest_neg.iterrows())):
            rowT_pos = row_pos.transpose()
            rowT_pos.to_csv(
                f"{PATH}_topic{topic}_pos{i}.csv",
                header=True)

            rowT_neg = row_neg.transpose()
            rowT_neg.to_csv(
                f"{PATH}_topic{topic}_neg{i}.csv",
                header=True)

