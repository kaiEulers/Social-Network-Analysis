"""
@author: kaisoon
"""
import pandas as pd, numpy as np
import seaborn as sns, matplotlib.pyplot as plt
import time as tm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import PowerTransformer
import importlib
import colourPals as cp
importlib.reload(cp)


def analyse(DATA, TAIL_THRES, SENTIDIFF_STD, TRANSFORM_METHOD='yeo-johnson', PLOT=False):
    """
    :param DATA: is a DataFrame with columns 'Person, Topic, and 'Speech'.
    :param TAIL_THRES: thresold at which data above will be removed
    :param SENTIDIFF_STD: sentiment difference that is above this normalised standard deviation with be removed
    :param TRANSFORM_METHOD: choose between 'yeo-johnson' or 'box-cox'. 'box-cox' only works for positive data.
    :param PLOT: option to visualise distribution transformation and threshold to remove outliers
    :return:
    """
    # ================================================================================
    # ----- FOR DEBUGGING
    # PATH = f"results/"

    # PARAMETERS
    # METHOD = 'nmf'
    # TAIL_THRES = 0.1
    # SENTIDIFF_STD = 1.5
    # TRANSFORM_METHOD = 'yeo-johnson'
    # PLOT = True
    # DATA = pd.read_csv(f"{PATH}ssm_results_{METHOD}.csv")
    # ================================================================================
    # ----- Sentiment Analysis
    startTime = tm.perf_counter()
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
        if i%20 == 0:
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
        for j, row_j in results[:i + 1].iterrows():
            p2 = row_j['Person']
            t2 = row_j['Topic']
            s2 = row_j['Senti_comp']
            # Compared speeches cannot be from the same person
            # Compared speeches must be of the same topic
            if p1 != p2 and t1 == t2 and s1*s2 > 0:
                sentiDiff.append(abs(s1 - s2))

        # Print progress
        if i%20 == 0:
            print(f"{i:{5}} of {len(results):{5}}\t{score}")

    # Organise sentiDiff in a dataframe
    sentiDiff = pd.DataFrame(sentiDiff, columns='sentiDiff'.split())
    sentiDiff.sort_values(by='sentiDiff', ascending=False, inplace=True)

    # ================================================================================
    # ----- Remove long tail of sentiDiff
    sentiDiff_noTail = sentiDiff[sentiDiff['sentiDiff'] < TAIL_THRES]

    # ----- Transform sentiDiff
    pt = PowerTransformer(method=TRANSFORM_METHOD)
    # Find optimal lambda value of Yeo-Johnson transform
    pt.fit(sentiDiff_noTail)
    pt_lambda = pt.lambdas_
    # Transform each column to a normal distribution
    sentiDiff_trans = pt.transform(sentiDiff_noTail)[:, 0]
    sentiDiff_noTail['sentiDiff_trans'] = sentiDiff_trans

    # ----- Compute threshold to remove speeches base on sentiment difference
    # Find index of outliers 2 std away in the neg direction
    sentiDiff_stats = sentiDiff_noTail.describe()

    # Compute sentiDiff threshold and index speeches to be removed
    sentiDiff_thres_trans = sentiDiff_stats['sentiDiff_trans'].loc['mean'] + SENTIDIFF_STD* \
                            sentiDiff_stats['sentiDiff_trans'].loc['std']
    remove_i = sentiDiff_noTail[
        sentiDiff_noTail['sentiDiff_trans'] > sentiDiff_thres_trans].index

    # Inverse transform the thresholds to get the untransformed threshold
    sentiDiff_thres = pt.inverse_transform(np.array([sentiDiff_thres_trans]).reshape(1, -1))[
        0, 0]

    # ----- Optional visualisation of tail truncation & distribution transformation and threshold to remove outliers
    if PLOT:
        FIG_SIZE = 3
        LABEL_SIZE = 8
        sns.set_style("darkgrid")
        sns.set_context("notebook")

        fig1, ax = plt.subplots(nrows=2, ncols=1, figsize=(FIG_SIZE*3, FIG_SIZE*2), dpi=300)
        sns.distplot(sentiDiff['sentiDiff'], ax=ax[0], kde=False, color=cp.cbPaired['blue'])
        ax[0].axvline(x=TAIL_THRES, c=cp.cbPaired['red'])
        ax[0].set_title(f"Removal of Tail above threshold")
        ax[0].set_xlabel(f"Sentiment Difference", fontsize=LABEL_SIZE)
        ax[0].set_ylabel(f"Frequency", fontsize=LABEL_SIZE)

        sns.distplot(sentiDiff_noTail['sentiDiff'], ax=ax[1], kde=False, color=cp.cbPaired['blue'])
        ax[1].set_xlabel(f"Sentiment Difference", fontsize=LABEL_SIZE)
        ax[1].set_ylabel(f"Frequency", fontsize=LABEL_SIZE)

        plt.tight_layout()
        plt.show()

        fig2, ax = plt.subplots(nrows=3, ncols=1, figsize=(FIG_SIZE*3, FIG_SIZE*3), dpi=300)
        sns.distplot(sentiDiff_noTail['sentiDiff'], ax=ax[0], color=cp.cbPaired['blue'])
        ax[0].axvline(x=sentiDiff_thres, c=cp.cbPaired['red'])
        ax[0].set_title(f"Removal of Outliers above threshold")
        ax[0].set_xlabel(f"Sentiment Difference", fontsize=LABEL_SIZE)
        ax[0].set_ylabel(f"Kernel Density", fontsize=LABEL_SIZE)

        sns.distplot(sentiDiff_noTail['sentiDiff_trans'], ax=ax[1], color=cp.cbPaired['purple'])
        ax[1].axvline(x=sentiDiff_thres_trans, c=cp.cbPaired['red'])
        ax[1].set_title(f"lambda = {round(pt_lambda[0], 3)}")
        ax[1].set_xlabel(f"Sentiment Difference Transformed", fontsize=LABEL_SIZE)
        ax[1].set_ylabel(f"Kernel Density", fontsize=LABEL_SIZE)

        qqplot(sentiDiff_noTail['sentiDiff_trans'], ax=ax[2], line='s',
               color=cp.cbPaired['purple'])

        plt.tight_layout()
        plt.show()

        fig1.savefig(f"results/ssm_sentiDiff_tailTruncation.png")
        fig2.savefig(f"results/ssm_sentiDiff_analysis.png")

    # ================================================================================

    print(f"{round(len(remove_i)/len(sentiDiff_noTail)*100, 1)}% of agreements will be removed")
    dur = tm.gmtime(tm.perf_counter() - startTime)
    print(f"\nSentiment analysis complete!\nAnalysis took {dur.tm_sec}s")

    # ================================================================================
    # ----- FOR DEBUGING
    # Save results
    # results.to_csv(f"{PATH}ssm_results_{METHOD}_senti.csv", index=False)
    # sentiDiff.to_csv(f"{PATH}stats/ssm_sentiDiff_{METHOD}.csv")
    # with open(f"{PATH}stats/ssm_sentiDiffThres_{METHOD}.txt", "w") as file:
    #     file.write(str(sentiDiff_thres))
    # ================================================================================

    return results, sentiDiff_thres, sentiDiff
