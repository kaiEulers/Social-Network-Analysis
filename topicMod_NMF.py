"""
@author: kaisoon
"""
# -----Imports
import numpy as np
from scipy import stats as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


# =====================================================================================
# Topic Modelling using Non-negative Matrix Factorisation(NMF)
# Modelling topics for speeches made in the month of 2017-12
DATE = '2017-12'
PATH = f"results/{DATE}/"
data = pd.read_csv(f"data/ssm_{DATE}_cleaned.csv")
# TODO: Need to model more topics!
nTOPICS = 3

# Instantiate Tfidf model
tfidf = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
# Instantiate NMF model
nmf_model = NMF(n_components=nTOPICS)

# Create document term matrix with tfidf model
dtm = tfidf.fit_transform(data['Speech'])

# Extract topics from speeches using NMF
# Apply non-negative matrix factorisation on the document term matrix
nmf_model.fit(dtm)
# nmf_model.transform() returns a matrix with coefficients that shows how much each document belongs to each topic
topicResults_nmf = nmf_model.transform(dtm)

# Store top w words in dataFrame topics_nmf
# Number of words that describes topic
w = 15
topics_nmf = pd.DataFrame()
for index,topic in enumerate(nmf_model.components_):
    # Negating an array causes the highest value to be the lowest value and vice versa
    topWordsIndex = (-topic).argsort()[:w]
    topics_nmf = topics_nmf.append(pd.Series([tfidf.get_feature_names()[i] for i in topWordsIndex]), ignore_index=True)
topics_nmf = topics_nmf.transpose()


# =====================================================================================
# Determine difference in top-two coefficient such that speeches can be class as invovling have one or more topics
# List all coefficient difference of the top-two topics
coeffSorted = [sorted(row) for row in topicResults_nmf]
coeffDiff = [row[-1] - row[-2] for row in coeffSorted]
# Compute percentile
percentile = [v/max(coeffDiff) for v in coeffDiff]

# Frame in a dataFrame
coeffDiff = pd.DataFrame(coeffDiff, columns='Topic_Coeff_Diff'.split())
coeffDiff['Percentile'] = percentile

# Save coeffDiff
coeffDiff.sort_values(by='Percentile', ascending=False, inplace=True)
coeffDiff.to_csv(f"{PATH}/distributions/ssm_{DATE}_topicCoeffDiff.csv", index=False)


# =====================================================================================
# Assign topic/s only to speeches with significantly large top-two topic coeff
# TODO: How do I determine this threshold? WE WANT COEFF DIFFERENCE THAT ARE HIGH!
COEFFDIFF_PERC_THRES = 0.1
# Filter out all data coeffDiff that is below the threshold
coeffDiff_sig = coeffDiff[coeffDiff['Percentile'] > COEFFDIFF_PERC_THRES]
results = data.iloc[coeffDiff_sig.index]

# Assign topic/s to each speech
topicAssigned_nmf = [topicResults_nmf[i].argmax() for i,row in coeffDiff_sig.iterrows()]

# Concat assigned topics
speeches = results['Speech']
results = results.drop('Speech', axis=1)
results["Topic_nmf"] = topicAssigned_nmf
results["Speech"] = speeches
results = results.sort_values(by='Speech_id')

# Save results
results.to_csv(f"{PATH}ssm_{DATE}_results_NMF.csv", index=False)


# =====================================================================================
# Analyse results from NMF topic modelling
# Compute topicCount and percentage of each topic

topicCount = {t: topicAssigned_nmf.count(t) for t in range(nTOPICS)}
percentage = {t: round(topicCount[t]/sum(topicCount.values()), 2) for t in range(nTOPICS)}

topics_nmf.loc['topicCount'] = topicCount
topics_nmf.loc['percentage'] = percentage
# Save analysis of topics
topics_nmf.to_csv(f"{PATH}ssm_{DATE}_topicsNMF.csv")

print(topics_nmf)
# Print percentage of speeches removed by threshold
print(f"\nPercentage of speeches removed by coeffDiff threshold: {round((len(coeffDiff)-len(coeffDiff_sig))/len(coeffDiff), 4)*100}%")