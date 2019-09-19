"""
@author: kaisoon
"""
import pandas as pd
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def sentiAnal(DATA, SENTIDIFF_PERC_THRES):
    """
    :param DATA: is a DataFrame with columns 'Person, Topic, and 'Speech'.
    :param SENTIDIFF_PERC_THRES:
    :return:
    """
    # ================================================================================
    # ----- FOR DEBUGGING
    # PATH = f"results/"

    # PARAMETERS
    # TODO: How do I determine the sentiment diff threshold? WE WANT SENTIDIFF THAT ARE LOW!
    # DATA = pd.read_csv(f"{PATH}ssm_results_nmf.csv")
    # SENTIDIFF_PERC_THRES = 0.1
    # ================================================================================
    # ----- Sentiment Analysis
    startTime = time.time()
    sid = SentimentIntensityAnalyzer()
    senti = pd.DataFrame(columns="pos neu neg compound".split())
    # ----- Analyse sentiment of all speeches
    print("Analysing sentiment...")
    # If topic is not nan, analyse sentiment. Otherwise, sentiment is also nan.
    for i in DATA.index:
        speech = DATA.loc[i]["Speech"]
        score = sid.polarity_scores(speech)
        senti.loc[i] = score

        # Print progress
        if i % 20 == 0:
            print(f"{i:{5}} of {len(DATA):{5}}\t{score}")

    # ----- Concat sentiments with results
    speeches = DATA['Speech']
    results = DATA.drop('Speech', axis=1)
    results['Senti_pos'] = senti['pos']
    results['Senti_neu'] = senti['neu']
    results['Senti_neg'] = senti['neg']
    results['Senti_comp'] = senti['compound']
    results['Speech'] = speeches
    results.sort_index(inplace=True)

    # ================================================================================
    # ----- Generate sentiDiff of all speeches with the same topic
    print("\nComputing sentiment difference...")
    sentiDiff = []
    for i, row_i in results.iterrows():
        p1 = row_i['Person']
        t1 = row_i['Topic']
        s1 = row_i['Senti_comp']
        for j, row_j in results[:i+1].iterrows():
            p2 = row_j['Person']
            t2 = row_j['Topic']
            s2 = row_j['Senti_comp']
            # Compared speeches cannot be from the same person
            # Compared speeches must be of the same topic
            if p1 != p2 and t1 == t2 and s1*s2 > 0:
                sentiDiff.append(abs(s1-s2))

        # Print progress
        if i % 20 == 0:
            print(f"{i:{5}} of {len(results):{5}}\t{score}")

    # Organise sentiDiff in a dataframe
    sentiDiff = pd.DataFrame(sentiDiff, columns='sentiDiff'.split())
    sentiDiff.sort_values(by='sentiDiff', ascending=False, inplace=True)
    # Compute threshold at which to remove speeches with high sentiment difference
    # TODO: Need to change the way the threshold is computed here?
    sentiDiff_thres = round(len(sentiDiff)*SENTIDIFF_PERC_THRES)

    print(f"Sentiment analysis complete!\nAnalysis took {round(time.time()-startTime, 2)}s")

    # ================================================================================
    # ----- FOR DEBUGING
    # Save results
    # results.to_csv(f"{PATH}ssm_results_lda_senti.csv", index=False)
    # sentiDiff.to_csv(f"{PATH}stats/ssm_sentiDiff.csv", index=False)
    # with open(f"{PATH}stats/ssm_sentiDiffThres.txt", "w") as file:
    #     file.write(str(sentiDiff_thres))
    # ================================================================================

    return results, sentiDiff_thres, sentiDiff
