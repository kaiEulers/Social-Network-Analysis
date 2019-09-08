"""
@author: kaisoon
"""
# Imports & Functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer


#%%---------- Sentiment Analysis
startTime = time.time()
DATE = '2017-12'
FILE_NAME = f"ssm_{DATE}_results_NMF.csv"
results = pd.read_csv(f"results/{FILE_NAME}")


sid = SentimentIntensityAnalyzer()
senti = pd.DataFrame(columns="pos neu neg compound".split())
# ----- Analyse sentiment of all speeches
print("Analysing sentiment...")
# If topic is not nan, analyse sentiment. Otherwise, sentiment is also nan.
for i in results.index:
    speech = results.loc[i]["Speech"]
    score = sid.polarity_scores(speech)
    senti.loc[i] = score

    # Print progress
    if i % 20 == 0:
        print(f"{i:{5}} of {len(results):{5}}\t{score}")
print(f"Sentiment analysis complete! Analysis took {time.time()-startTime}s")


# ----- Concat sentiments with results
speeches = results['Speech']
results = results.drop('Speech', axis=1)
results['Senti_pos'] = senti['pos']
results['Senti_neu'] = senti['neu']
results['Senti_neg'] = senti['neg']
results['Senti_comp'] = senti['compound']
results['Speech'] = speeches

# Save results
results.to_csv(f"results/{FILE_NAME.replace('NMF', 'NMF_senti')}", index=False)


#%% Analyse Results
sns.set_style("darkgrid")
sns.set_context("notebook")

# ----- Plot histograms of compound, pos, neu, and neg sentiment
fig = plt.figure(dpi=300)

plt.subplot(2, 2, 1)
ax1 = sns.distplot(senti['pos'], bins=30, kde=True, norm_hist=False)

plt.subplot(2, 2, 2)
ax2 = sns.distplot(senti['neg'], bins=30, kde=True, norm_hist=False)

plt.subplot(2, 2, 3)
ax3 = sns.distplot(senti['neu'], bins=30, kde=True, norm_hist=False)

plt.subplot(2, 2, 4)
ax4 = sns.distplot(senti['compound'], bins=30, kde=False, norm_hist=False)
# ax4.set_ylim([0, 8])

plt.show()

# Members who do not support SSM
# Russell Broadbent, Keith Pitt, David Littleproud, Bob Katter
# Speech/es by Katter 216-219
# Speech/es by Pitt 188
# Speech/es by Littleproud 258

# True Positive - #216 #198
# False Positive - 11
# True Negative - #217 #218 #219

