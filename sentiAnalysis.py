#%% Imports & Functions
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer

#%%---------- Sentiment Analysis
FILE_NAME = "ssm_results_NMF_2017-12.csv"
results = pd.read_csv(f"data/results/{FILE_NAME}")

sid = SentimentIntensityAnalyzer()

senti = pd.DataFrame(columns="pos neu neg compound".split())
# ----- Analyse sentiment of all speeches
for i in results.index:
    # Analyse Sentiment
    score = sid.polarity_scores(results.iloc[i]["Speech"])
    senti = senti.append(score, ignore_index=True)

    # Print progress
    if i % 20 == 0:
        print(f"{i} of {len(results)}\t{score}")

# ----- Concat sentiments with results
speeches = results['Speech']
results = results.drop('Speech', axis=1)
results['Senti_pos'] = senti['pos']
results['Senti_neu'] = senti['neu']
results['Senti_neg'] = senti['neg']
results['Senti_comp'] = senti['compound']
results['Speech'] = speeches

# Save results
results.to_csv(f"data/results/{re.sub('NMF', 'NMF_senti', FILE_NAME)}", index=False)


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

