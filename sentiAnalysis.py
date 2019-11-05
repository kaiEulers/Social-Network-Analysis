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
import kaiFunctions as mf
importlib.reload(cp)
importlib.reload(mf)


def analyse(DATA):
    """
    :param DATA: is a DataFrame with columns 'Person, Topic, and 'Speech'.
    :param TAIL_THRES: thresold at which text above will be removed
    :param SENTIDIFF_STD: sentiment difference that is above this normalised standard deviation with be removed
    :param TRANSFORM_METHOD: choose between 'yeo-johnson' or 'box-cox'. 'box-cox' only works for positive text.
    :param PLOT: option to visualise distribution transformation and threshold to remove outliers
    :return:
    """
    # ================================================================================
    # ----- FOR DEBUGGING
    PATH = f"results/"

    # PARAMETERS
    METHOD = 'nmf'
    # TAIL_THRES = 0.4
    # SENTIDIFF_STD = 1.5
    # TRANSFORM_METHOD = 'box-cox'
    # PLOT = True
    # text = pd.read_csv(f"{PATH}ssm_results_{METHOD}.csv")
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
    speeches = DATA[['Speech', 'Speech_lemma']]
    results = DATA.drop(['Speech', 'Speech_lemma'], axis=1)
    results['Senti_pos'] = senti['pos']
    results['Senti_neu'] = senti['neu']
    results['Senti_neg'] = senti['neg']
    results['Senti_comp'] = senti['compound']
    results = pd.concat([results, speeches], axis=1)
    results.sort_index(inplace=True)

    # ================================================================================
    # # ----- Generate sentiDiff of all speeches with the same topic
    # print("\nComputing sentiment difference...")
    # sentiDiff = []
    # for i, row_i in results.iterrows():
    #     p_i = row_i['Person']
    #     t_i = row_i['Topic']
    #     s_i = row_i['Senti_comp']
    #     for j, row_j in results[:i + 1].iterrows():
    #         p_j = row_j['Person']
    #         t_j = row_j['Topic']
    #         s_j = row_j['Senti_comp']
    #         # Compared speeches cannot be from the same person
    #         # Compared speeches must be of the same topic
    #         if p_i != p_j and t_i == t_j and s_i*s_j > 0:
    #             sentiDiff.append(abs(s_i - s_j))
    #
    #         # Print progress...
    #         if (i%50 == 0) and (j%50 == 0):
    #             print(f"{i:{5}},{j:{5}} of {len(results):{5}}\t{score}")

    # # Organise sentiDiff in a dataframe
    # sentiDiff = pd.Series(sentiDiff)
    # sentiDiff.sort_values(ascending=False, inplace=True)
    #
    # # ================================================================================
    # # ----- Remove long tail of sentiDiff
    # sentiDiff_withTail = sentiDiff
    # if TAIL_THRES != None:
    #     sentiDiff = sentiDiff[sentiDiff <= TAIL_THRES]

    # sentiDiff_noOutliers, sentiDiff_thres, sentiDiff_trans = kf.removeOutliers(sentiDiff, SENTIDIFF_STD, METHOD=TRANSFORM_METHOD)
    #
    # # ----- Optional visualisation of tail truncation & distribution transformation and threshold to remove outliers
    # if PLOT:
    #     FIG_SIZE = 3
    #     LABEL_SIZE = 8
    #     sns.set_style("darkgrid")
    #     sns.set_context("notebook")
    #
    #     fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(FIG_SIZE*3, FIG_SIZE*3), dpi=300)
    #     # Plot sentiDiff before transformation
    #     sns.distplot(sentiDiff, ax=ax[0], color=cp.cbPaired['blue'])
    #     ax[0].axvline(x=sentiDiff_thres, c=cp.cbPaired['red'])
    #     ax[0].set_title(f"Outliers removed above {SENTIDIFF_STD}std, tail truncated at {TAIL_THRES}")
    #     ax[0].set_xlabel(f"Sentiment Difference", fontsize=LABEL_SIZE)
    #     ax[0].set_ylabel(f"Kernel Density", fontsize=LABEL_SIZE)
    #     # Plot sentiDiff after transformation
    #     sns.distplot(sentiDiff_trans, ax=ax[1], color=cp.cbPaired['purple'])
    #     ax[1].axvline(x=SENTIDIFF_STD, c=cp.cbPaired['red'])
    #     ax[1].set_xlabel(f"Sentiment Difference Transformed", fontsize=LABEL_SIZE)
    #     ax[1].set_ylabel(f"Kernel Density", fontsize=LABEL_SIZE)
    #     # Plot qqplot of sentiDiff after transformation
    #     qqplot(sentiDiff_trans, ax=ax[2], line='s',color=cp.cbPaired['purple'])
    #
    #     plt.tight_layout()
    #     plt.show()
    #     fig.savefig(f"results/stats/ssm_sentiDiff_analysis.png")

    # ================================================================================

    # print(f"{round((1 - len(sentiDiff_noOutliers)/len(sentiDiff_withTail))*100, 2)}% of agreements will be removed")
    dur = tm.gmtime(tm.perf_counter() - startTime)
    print(f"\nSentiment analysis complete! Analysis took {dur.tm_sec}s")

    # ================================================================================
    # ----- FOR DEBUGING
    # Save results
    # results.to_csv(f"{PATH}ssm_results_{METHOD}_senti.csv", index=False)
    # sentiDiff.to_csv(f"{PATH}stats/ssm_sentiDiff_{METHOD}.csv", header=False)
    # ================================================================================

    return results
    # return results, sentiDiff_thres
