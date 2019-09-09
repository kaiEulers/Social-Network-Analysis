"""
@author: kaisoon
"""
# Imports & Functions
import numpy as np
import pandas as pd
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# =====================================================================================
# Sentiment Analysis
startTime = time.time()
DATE = '2017-12'
PATH = f"results/{DATE}/"
results = pd.read_csv(f"{PATH}ssm_{DATE}_results_NMF.csv")

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

# ----- Concat sentiments with results
speeches = results['Speech']
results = results.drop('Speech', axis=1)
results['Senti_pos'] = senti['pos']
results['Senti_neu'] = senti['neu']
results['Senti_neg'] = senti['neg']
results['Senti_comp'] = senti['compound']
results['Speech'] = speeches
# Save results
results.to_csv(f"{PATH}ssm_{DATE}_results_NMF_senti.csv", index=False)


# =====================================================================================
# Generate sentiDiff of all speeches with the same topic
sentiDiff = []
for i in range(len(results)):
    row_i = results.iloc[i]
    p1 = row_i['Person']
    t1 = row_i['Topic_nmf']
    s1 = row_i['Senti_comp']
    for j in range(i+1, len(results)):
        row_j = results.iloc[j]
        p2 = row_j['Person']
        t2 = row_j['Topic_nmf']
        s2 = row_j['Senti_comp']
        # Compared speeches cannot be from the same person
        # Compared speeches must be of the same topic
        if p1 != p2 and t1 == t2 and s1*s2 > 0:
            sentiDiff.append(abs(s1-s2))

    # Print progress
    if i % 20 == 0 :
        print(f"{i:{5}} of {len(results):{5}}\t{score}")

# Compute percentile
percentile = [sd/max(sentiDiff) for sd in sentiDiff]

# Save sentiDiff
sentiDiff = pd.DataFrame(sentiDiff, columns='Senti_Diff'.split())
sentiDiff['Percentile'] = percentile
sentiDiff = sentiDiff.sort_values(by='Percentile', ascending=False)
sentiDiff.to_csv(f"{PATH}distributions/ssm_{DATE}_sentiDiff.csv", index=False)

print(f"Sentiment analysis complete! Analysis took {time.time()-startTime}s")