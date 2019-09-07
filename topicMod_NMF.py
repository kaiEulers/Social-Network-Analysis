"""
@author: kaisoon
"""
# -----Imports
import numpy as np
from scipy import stats as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


#%%---------- Topic Modelling using Non-negative Matrix Factorisation(NMF)
# ----- Modelling topics for speeches made in the month of 2017-12
DATE = '2017-12'
FILE_NAME = f"ssm_{DATE}_cleaned.csv"
data = pd.read_csv(f"data/{FILE_NAME}")
# TODO: Need to model more topics!
nTOPICS = 3

# Instantiate Tfidf model
tfidf = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
# Instantiate NMF model
nmf_model = NMF(n_components=nTOPICS)

print("Modelling topic with NMF...")
# ----- Create document term matrix with tfidf model
dtm = tfidf.fit_transform(data['Speech'])

# ----- Extract topics from speeches using NMF
# Apply non-negative matrix factorisation on the document term matrix
nmf_model.fit(dtm)
# nmf_model.transform() returns a matrix with coefficients that shows how much each document belongs to each topic
topic_results_nmf = nmf_model.transform(dtm)

# Store top w words in dataFrame topics_nmf
# Number of words that describes topic
w = 15
topics_nmf = pd.DataFrame()
for index,topic in enumerate(nmf_model.components_):
    # Negating an array causes the highest value to be the lowest value and vice versa
    topWordsIndex = (-topic).argsort()[:w]
    topics_nmf = topics_nmf.append(pd.Series([tfidf.get_feature_names()[i] for i in topWordsIndex]), ignore_index=True)
topics_nmf = topics_nmf.transpose()
print("NMF topic modelling complete!")


# ----- Determine difference in coefficient such that speeches can be class as invovling have one or more topics
print("\nComputing difference in top-two topic coefficients...")
# Rank the topics that each speech is about
rank = []
for row in topic_results_nmf:
    rank.append(row.argsort())

# Find difference between coefficients
# Small difference means that speech involves 2 or more topics!
topicCoeff = pd.Series(np.zeros(len(topic_results_nmf)))

for i in range(len(topic_results_nmf)):
    # Extract top coefficients
    # max[0] contains the highest coefficient
    # max[1] contains the 2nd highest coefficient ...
    max = pd.Series(np.zeros(nTOPICS))
    for j in range(nTOPICS):
        max[j] = topic_results_nmf[i][rank[i][-j-1]]

    # Compute difference between highest and 2nd highest coeff
    topicCoeff[i] = max[0] - max[1]

print("Computing of difference complete!")


# ----- Determine threshold to class speeches as having more than one topic
SIG_LEVEL = 0.1
CoeffStats = topicCoeff.describe()
# Depending on the threshold of difference between coefficients, speeeches can be assigned to more than one topics
# thres_12 is the difference in coefficient between the highest and 2nd highest topic for speech to be considered to be about both topics. If the different is this value or less, speech will be about both topics.
# thres_12 is statistically determined, assuming that the difference in topic coefficient exhibits a normal distribution
# TODO: Empirically determine threshold??? (ie. sort all topicCoeff and find the difference at 10% of the data)
# TODO: Try finding the difference relative to min coeff. Want to remove small differences!
coeffDiff_thres = CoeffStats['std'] * st.norm.ppf(SIG_LEVEL) + CoeffStats['mean']


# ----- Assign topic/s to each speech
topic_assigned_nmf = pd.Series(np.empty(len(topic_results_nmf)))
topic_assigned_nmf[:] = np.nan

for i in range(len(topic_results_nmf)):
    if topicCoeff[i] < coeffDiff_thres:
        topic_assigned_nmf[i] = np.nan
    else:
        topic_assigned_nmf[i] = topic_results_nmf[i].argmax()


# ----- Concat results
speeches = data['Speech']
results = data.drop('Speech', axis=1)
results["Topic_nmf"] = topic_assigned_nmf
results["Speech"] = speeches

# Remove rows that contains two or more topics
results = results.dropna().reset_index(drop=True)

# Save results
results.to_csv(f"results/{FILE_NAME.replace('cleaned', 'results_NMF')}", index=False)


# ----- Analyse results from NMF topic modelling
# Check percentage of speechCount of each topic
# Expand topic_assigned into a list of all topics assigned
topicList = pd.Series()
for row in topic_assigned_nmf:
    topicList = topicList.append(pd.Series(row), ignore_index=True)

# Count total number of topics all speeches are involved in
analysis_nmf = pd.DataFrame(topicList.value_counts().sort_index(), columns='SpeechCount'.split())
analysis_nmf["Percentage"] = [round(cnt/sum(analysis_nmf["SpeechCount"]), 2) for cnt in analysis_nmf["SpeechCount"]]

# Concat analysis_nmf to topics_nmf
topics_nmf = topics_nmf.append(analysis_nmf["Percentage"])
topics_nmf = topics_nmf.append(analysis_nmf["SpeechCount"])
print("\n", topics_nmf)

# Percentage of speeches similar top two topic coefficients
num_thres = len(topicCoeff[topicCoeff < coeffDiff_thres])
num_total = len(topic_results_nmf)
diff_perc = (num_thres) / num_total

print(f"\n{round(diff_perc * 100, 2)}% of speeches have similar coefficients for their top two topics")

# Save topics and analysis_nmf
topics_nmf.to_csv(f"results/{FILE_NAME.replace('cleaned', 'topics_NMF')}")


#%% Plots
sns.set_style("darkgrid")
sns.set_context("notebook")

# ----- Plot distribution of coeff difference
fig = plt.figure(dpi=300)

sns.distplot(diff['diff_12'], bins=5, kde=True, norm_hist=False)
plt.show()

fig.savefig(f"results/{FILE_NAME.replace('cleaned.csv', 'coeffDiff.png')}", dpi=300)
