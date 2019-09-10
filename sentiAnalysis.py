"""
@author: kaisoon
"""
def sentiAn(DATA, SENTIDIFF_PERC_THRES):
    import pandas as pd
    import time
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    # =====================================================================================
    # ----- FOR DEBUGGING
    DATE = '2017-12'
    PATH = f"results/{DATE}/"

    # PARAMETERS
    # TODO: How do I determine the sentiment diff threshold? WE WANT SENTI DIFF THAT ARE LOW!
    # DATA = pd.read_csv(f"{PATH}ssm_{DATE}_results_NMF.csv")
    # SENTIDIFF_PERC_THRES = 0.2


    # =====================================================================================
    # Sentiment Analysis
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
    DATA = DATA.drop('Speech', axis=1)
    DATA['Senti_pos'] = senti['pos']
    DATA['Senti_neu'] = senti['neu']
    DATA['Senti_neg'] = senti['neg']
    DATA['Senti_comp'] = senti['compound']
    DATA['Speech'] = speeches


    # =====================================================================================
    # Generate sentiDiff of all speeches with the same topic
    print("\nComputing sentiment difference...")
    sentiDiff = []
    for i in range(len(DATA)):
        row_i = DATA.iloc[i]
        p1 = row_i['Person']
        t1 = row_i['Topic_nmf']
        s1 = row_i['Senti_comp']
        for j in range(i+1, len(DATA)):
            row_j = DATA.iloc[j]
            p2 = row_j['Person']
            t2 = row_j['Topic_nmf']
            s2 = row_j['Senti_comp']
            # Compared speeches cannot be from the same person
            # Compared speeches must be of the same topic
            if p1 != p2 and t1 == t2 and s1*s2 > 0:
                sentiDiff.append(abs(s1-s2))

        # Print progress
        if i % 20 == 0 :
            print(f"{i:{5}} of {len(DATA):{5}}\t{score}")

    # Organise results in dataframe
    sentiDiff = pd.DataFrame(sentiDiff, columns='Senti_Diff'.split())
    sentiDiff.sort_values(by='Senti_Diff', ascending=False, inplace=True)
    # Compute threshold at which to remove speeches with high sentiment
    sentiDiff_thres = round(len(sentiDiff)*SENTIDIFF_PERC_THRES)
    # Only take data below the threshold
    sentiDiff_sig = sentiDiff[sentiDiff_thres:]

    print(f"Sentiment analysis complete!\nAnalysis took {round(time.time()-startTime, 2)}s")


    # =====================================================================================
    # Save results
    DATA.to_csv(f"{PATH}ssm_{DATE}_results_NMF_senti.csv", index=False)
    # Save sentiDiff
    sentiDiff.to_csv(f"{PATH}distributions/ssm_{DATE}_sentiDiff.csv", index=False)
    # Save sentiDiff_thres
    with open(f"{PATH}distributions/ssm_{DATE}_sentiDiffThres.txt", "w") as file:
        file.write(str(sentiDiff_thres))

    return DATA, sentiDiff_thres, sentiDiff_sig
