"""
@author: kaisoon
"""
import os
import re
from copy import copy
from datetime import datetime as dt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import spacy
nlp = spacy.load('en_core_web_sm')
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from scipy import stats
import itertools as iter
idx = pd.IndexSlice

from importlib import reload
import processText
import constructProfile
import constructGraph
import drawGraph
import concurrent.futures
import kaiFunctions as kf
from kaiFunctions import save_pickle, load_pickle, log
import colourPals as cp

reload(kf)
reload(cp)

TF_DATES = {
    'All'              : [(2002, 2, 12), (2019, 4, 11)],
    # '40th'             : [(2002, 2, 12), (2004, 8, 31)],
    # '43rd'             : [(2010, 9, 28), (2013, 8, 5)],
    # '44th'             : [(2013, 11, 12), (2016, 5, 9)],
    # '45th'             : [(2016, 8, 30), (2019, 4, 11)],
    # '45thPrePleb'      : [(2016, 8, 30), (2017, 11, 14)],
    # '45thPostPleb'     : [(2017, 11, 15), (2017, 12, 7)],
    'Howard'           : [(2002, 2, 12), (2004, 8, 31)],
    'Gillard'          : [(2010, 9, 28), (2013, 6, 26)],
    'Abbott'           : [(2013, 11, 12), (2015, 9, 14)],
    'Turnbull'         : [(2015, 9, 15), (2017, 12, 7)],
    'Turnbull-PreSurvey' : [(2015, 9, 15), (2017, 11, 14)],
    'Turnbull-PostSurvey': [(2017, 11, 15), (2017, 12, 7)],

    # # 40th Parliament Sittings
    # '2004-06-16 to 06-17': [(2004, 6, 16), (2004, 6, 17)],
    # '2004-06-24': [(2004, 6, 24)],
    # # 43rd Parliament Sittings
    # '2012-02-27 to 02-28': [(2012, 2, 27), (2012, 2, 28)],
    # '2012-06-18': [(2012, 6, 18)],
    # '2012-06-25': [(2012, 6, 25)],
    # '2012-08-20': [(2012, 8, 20)],
    # '2012-09-10': [(2012, 9, 10)],
    # '2013-03-18': [(2013, 3, 18)],
    # '2013-05-27': [(2013, 5, 27)],
    # '2013-06-03': [(2013, 6, 3)],
    # '2013-06-17': [(2013, 6, 17)],
    # # 44th Parliament Sittings
    # '2015-06-01': [(2015, 6, 1)],
    # '2015-06-15': [(2015, 6, 15)],
    # '2015-06-22': [(2015, 6, 22)],
    # '2015-08-12': [(2015, 8, 12)],
    # '2015-08-17': [(2015, 8, 17)],
    # '2015-09-07': [(2015, 9, 7)],
    # '2015-09-14': [(2015, 9, 14)],
    # '2015-10-12': [(2015, 10, 12)],
    # '2015-10-19': [(2015, 10, 19)],
    # '2015-11-09': [(2015, 11, 9)],
    # '2015-11-23': [(2015, 11, 23)],
    # '2016-02-08': [(2016, 2, 8)],
    # '2016-05-02': [(2016, 5, 2)],
    # # 45th Parliament Sittings
    # '2016-09-12 to 09-14': [(2016, 9, 12), (2016, 9, 14)],
    # '2016-10-11 to 10-13': [(2016, 10, 11), (2016, 10, 13)],
    # '2016-10-19 to 10-20': [(2016, 10, 19), (2016, 10, 20)],
    # '2016-11-21': [(2016, 11, 21)],
    # '2017-12-04 to 12-07': [(2017, 12, 4), (2017, 12, 7)],
}
TF_LABELS = {
    'All'         : 'All Terms',
    '40th'        : '40th Parliament',
    '43rd'        : '43rd Parliament',
    '44th'        : '44th Parliament',
    '45th'        : '45th Parliament',
    '45thPrePleb' : '45th Parliament, Pre-Plebiscite',
    '45thPostPleb': '45th Parliament, Post-Plebiscite',
    'Howard'  : 'Howard Goverment',
    'Gillard' : 'Giillard Goverment',
    'Abbott'  : 'Abbott Goverment',
    'Turnbull': 'Turnbull Goverment',
    'Turnbull-PreSurvey': 'Turnbull Goverment\nPre Postal Survey',
    'Turnbull-PostSurvey': 'Turnbull Goverment\nPost Postal Survey',
}
TF_LABELS = pd.Series(TF_LABELS)
TIME_FRAMES = {
    # 'terms'    : pd.Categorical('40th 43rd 44th 45th'.split(), ordered=True),
    'termsPleb': pd.Categorical('40th 43rd 44th 45thPrePleb 45thPostPleb'.split(), ordered=True),
    'termsPM': pd.Categorical('Howard Gillard Abbott Turnbull-PreSurvey Turnbull-PostSurvey'.split(), ordered=True),
}
TIME_FRAMES['termsPleb'].reorder_categories('40th 43rd 44th 45thPrePleb 45thPostPleb'.split(), inplace=True)
TIME_FRAMES['termsPM'].reorder_categories('Howard Gillard Abbott Turnbull-PreSurvey Turnbull-PostSurvey'.split(), inplace=True)

SYM = {
    'Party'   : {
        'label' : {
            'AG'  : "Australian Greens",
            'ALP' : "Australian Labor Party",
            'CA'  : "Central Alliance",
            'IND' : "Independent",
            'KAP' : "Katter's Australian Party",
            'LP'  : "Liberal Party of Australia",
            'Nats': "National Party of Australia",
        },
        'colour': {
            'AG'  : '#39B54A',
            'ALP' : '#DE3533',
            'CA'  : '#FF7400',
            'IND' : cp.cbSet1['brown'],
            'KAP' : cp.cbSet1['purple'],
            # 'LP'  : '#1B46A5',
            'LP'  : cp.sns['blue'],
            'Nats': '#FEF032',
        }
    },
    'PartyAgg': {
        'label' : {
            'Coalition'  : "Coalition",
            'Labor'     : "Australian Labor Party",
            'Greens'     : "Australian Greens",
            # 'Katter'     : "Katter's Australia Party",
            'Minor Party': "Minor Parties",
        },
        'colour': {
            # 'Coalition'  : '#1B46A5',
            'Coalition'  : '#0047AB',
            # 'Coalition'  : cp.sns['blue'],
            'Labor'     : '#DE3533',
            'Greens'     : '#39B54A',
            # 'Katter'     : "Katter's Australia Party",
            'Minor Party': cp.cbSet1['brown'],
        }
    },
    'Gender'  : {
        'label' : {'Female': 'Female', 'Male': 'Male', },
        'colour': {'Female': cp.cbSet2['pink'], 'Male': cp.cbSet2['blue'], },
    },
    'Metro'   : {
        'label' : {1: 'Zone 1', 2: 'Zone 2', 3: 'Zone 3', 4: 'Zone 4', },
        'colour': {1: cp.cbYlGn[0], 2: cp.cbYlGn[1],
                   3: cp.cbYlGn[2], 4: cp.cbYlGn[3], },
    },
    'Topic'   : {
        'label'      : {
            0 : "Love & Equality",
            1 : "Religious Freedom",
            2 : 'Institution of Marriage',
            7 : "Same-sex Discrimination",
            10: "Same-sex Parenting",
        },
        'colour'     : {
            0 : cp.cbPaired['red'],
            1 : cp.cbPaired['purple'],
            2 : cp.cbPaired['blue'],
            7 : cp.cbPaired['green'],
            10: cp.cbPaired['orange'],
        },
        'colourLight': {
            0 : cp.cbPaired['lightRed'],
            1 : cp.cbPaired['lightPurple'],
            2 : cp.cbPaired['lightBlue'],
            7 : cp.cbPaired['lightGreen'],
            10: cp.cbPaired['lightOrange'],
        },
        'marker'     : {0: '*', 1: 'P', 2: 's', 7: 'd', 10: 'p'}
    },
}
EMP_MAP = {
    'node_size'      : [1, 2],
    'node_alpha'     : [0.5, 1],
    'node_edgewidth' : [1, 8],
    'node_edgecolour': ['black', 'black'],

    'font_size'      : [1, 1.1],
    'font_colour'    : ['black', 'black'],
    'font_alpha'     : [0.5, 1],
    'font_weight'    : ['normal', 'bold'],

    'edge_colour'    : [cp.cbGrays[3], cp.cbGrays[4], cp.cbGrays[5], cp.cbGrays[6]],
    'edge_alpha'     : [0.1, 1],
    'edge_width'     : [1, 1.5, 2, 3],
}
G_STYLE = {
    'node_size'      : 400,
    'node_alpha'     : 0.85,
    'node_edgewidth' : 0.2,
    'node_edgecolour': 'black',

    'font_size'      : 5,
    'font_colour'    : 'black',
    'font_alpha'     : 1,
    'font_weight'    : 'normal',

    'edge_colour'    : cp.cbGrays[3],
    'edge_alpha'     : 0.5,
    'edge_width'     : 0.15,
}
assert len(G_STYLE) == len(EMP_MAP)

CLQ_PROFILE_COLOURS = {}
CLQ_PROFILE_COLOURS['Size'] = 'black'
for grp in 'Party Gender Metro'.split():
    CLQ_PROFILE_COLOURS.update(SYM[grp]['colour'])
topicColour = {}
for topic, colour in zip(SYM['Topic']['label'].values(), SYM['Topic']['colour'].values()):
    topicColour[topic] = colour
CLQ_PROFILE_COLOURS.update(topicColour)

# Select between 'para' or 'speech' resolution for analysis
RES = 'para'
nTOPICS = 11
COEFFMAX = {
    'All'         : 0,
    '40th'        : 0,
    '43rd'        : 0,
    '44th'        : 0,
    '45th'        : 0.25,
    '45thPrePleb' : 0.25,
    '45thPostPleb': 0.25,
    'Howard'      : 0,
    'Gillard'     : 0,
    'Abbott'      : 0,
    'Turnbull'    : 0,
    'Turnbull-PreSurvey' : 0.25,
    'Turnbull-PostSurvey': 0.25,
}

THRESHOLD = 0
SAVE_SPEECHES = 3
# Select between 'nmf' or 'lda' topic modelling
METHOD = 'nmf'
# Select between 'box-cox' or 'yeo-johnson' transformation method for outlier detection
TRANSFORM_METHOD = 'box-cox'
PLOT = False

PATH = f"results/resolution_{RES}/"
data = pd.read_csv(f"data/ssm_{RES}Res_rel_lemma.csv")

if nTOPICS == 11 and RES == 'para':
    TODROP_TOPICS = [3, 4, 5, 6, 8, 9]
elif nTOPICS == 13 and RES == 'para':
    TODROP_TOPICS = [3, 4, 5, 6, 8, 9, 11]
else:
    TODROP_TOPICS = []
nWORDS = 20

assert nTOPICS > 1, "Number of topics model must be at least 2."
assert METHOD == 'nmf' or 'lda', "METHOD must be 'nmf' or 'lda'"
assert RES == 'para' or 'speech', "RES must be 'para' or 'speech'"

# %% ========== Topic Modelling
reload(processText)

print(f"Modelling topics...")
topicMatrix, topics, topicAssigned = processText.topicModel(
    data['Speech_lemma'], nTOPICS, nWORDS, METHOD=METHOD
)
# Retrive highest coefficent of each topic
topicMatrix_series = pd.Series([row for row in topicMatrix])
coeffMax = topicMatrix_series.apply(max)

# Concat topicAssigned to results
speeches = data[['Speech', 'Speech_lemma']]
results_tm = data.drop(['Speech', 'Speech_lemma'], axis=1)
results_tm["Topic"] = topicAssigned.astype(int)
results_tm["CoeffMax"] = coeffMax
results_tm = pd.concat([results_tm, speeches], axis=1)
results_tm.sort_index(inplace=True)

assert data.index.equals(topicMatrix_series.index)
topics.to_csv(f"{PATH}ssm_{nTOPICS}topics_{RES}Res_{METHOD}.csv")
print(topics)

# ========== Sentiment Analysis
reload(processText)

print(f"Analysing sentiment...")
sentiScore = processText.sentiAnalysis(results_tm['Speech'])
# Concat sentiment to results
speeches = results_tm[['Speech', 'Speech_lemma']]
results_tm_senti = results_tm.drop(['Speech', 'Speech_lemma'], axis=1)
results_tm_senti['Senti_pos'] = sentiScore['pos']
results_tm_senti['Senti_neu'] = sentiScore['neu']
results_tm_senti['Senti_neg'] = sentiScore['neg']
results_tm_senti['Senti_comp'] = sentiScore['compound']
results_tm_senti = pd.concat([results_tm_senti, speeches], axis=1)
results_tm_senti.sort_index(inplace=True)
print(f"Sentiment computed!\n")

# ========== Compute Word Count
nlp = spacy.load('en_core_web_sm')


def countWords(text):
    doc = nlp(text)
    tokenList = [token for token in doc if not token.is_punct]
    return len(tokenList)


print(f"Counting words...")
wordCountList = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    resultList = executor.map(countWords, data['Speech'])
    for result in resultList:
        wordCountList.append(result)

# Concat wordCount to results
speeches = results_tm_senti[['Speech', 'Speech_lemma']]
results_tm_senti_wc = results_tm_senti.drop(['Speech', 'Speech_lemma'], axis=1)
results_tm_senti_wc['WordCount'] = wordCountList
results_tm_senti_wc = pd.concat([results_tm_senti_wc, speeches], axis=1)
results_tm_senti_wc.sort_index(inplace=True)
print(f"Word count computed!\n")

# Save top 3 speeches with the highest coeffMax for auditing
processText.auditText(results_tm_senti_wc, f"realityChecks/highestCoeffMax_{METHOD}", NUM=3)

# ========== Iteratively remove irrelevant topics
print(f"Removing irrelevant topics...")
results_rel = copy(results_tm_senti_wc)
# argsort() returns an array with index of the smallest to the largest element in an array
# Apply argsort() to all rows of coefficient in the topicMatrix_series
topicMatrix_argSorted = topicMatrix_series.apply(np.argsort)
# Apply sort() to all rows of coefficient in the topicMatrix_series
topicMatrix_sorted = topicMatrix_series.apply(np.sort)
for i in range(2, nTOPICS + 1):
    # Retrieve rows of results whose dominant topic is irrelevant
    topic_irrel = results_rel[results_rel['Topic'].isin(TODROP_TOPICS)]
    # Break out of loop if there are no rows of results whose dominant topic is irrelevant
    if len(topic_irrel) == 0:
        break
    # Find the next best topic from these rows whose dominant topic is irrelevant
    nextBest = pd.DataFrame({
        'Topic'   : [row[-i] for row in topicMatrix_argSorted.loc[topic_irrel.index]],
        'CoeffMax': [row[-i] for row in topicMatrix_sorted.loc[topic_irrel.index]],
    }, index=topic_irrel.index)
    # Update the result dataframe with these next best topic
    results_rel.update(nextBest)

# If there are any more rows left with results whose dominant topic is irrelevant, drop them.
irrelevant = results_rel[results_rel['Topic'].isin(TODROP_TOPICS)]
irrelevant['Topic'] = -1
results_rel.update(irrelevant)
print(f"Irrelevant topics re-assigned!\n")

# ==========Divide results with outliers into time-frames
resultsOutliers = kf.convert2datetime(results_rel)

resultsOutliers_dict = {}
for key, tupleList in TF_DATES.items():
    if len(tupleList) == 1:
        tup = tupleList[0]
        year, month, date = tup[0], tup[1], tup[2]
        resultsOutliers_dict[key] = resultsOutliers[resultsOutliers['Date'] == dt(year, month, date)]
    elif len(tupleList) == 2:
        tup0, tup1 = tupleList[0], tupleList[1]
        year0, month0, date0 = tup0[0], tup0[1], tup0[2]
        year1, month1, date1 = tup1[0], tup1[1], tup1[2]
        resultsOutliers_dict[key] = resultsOutliers[(resultsOutliers['Date'] >= dt(year0, month0, date0)) & (
                resultsOutliers['Date'] <= dt(year1, month1, date1))]
    else:
        raise ValueError(
            'Too many tuples in date list. Can only contain 1 or 2 tuples.')

save_pickle(resultsOutliers_dict, f"{PATH}ssm_resultsDictOutliers_{nTOPICS}topics_{METHOD}")

# %% ========== Construct graphs with varying COEFFMAX
COEFFMAX_STD_LIST = np.arange(-2, 2, 0.25)
COEFFMAX_STD_LIST = [-1.5, -1, -0.5, -0.25, 0, 0.25, 1, 1.5]
FIG_SIZE = 5
NODE_SIZE = 300
EGDE_WIDTH = 0.2
GROUPBY = 'party'
pltconfig = {'nrows': 4, 'ncols': 4}

results_dict = {}
for tf in TF_DATES:
    fig = plt.figure(figsize=(FIG_SIZE*1.5*pltconfig['ncols'], FIG_SIZE*pltconfig['nrows']))
    for k, coeffMax in zip([0, 1, 2, 3, 8, 9, 10, 11], COEFFMAX_STD_LIST):
    # for k, coeffMax in enumerate(COEFFMAX_STD_LIST):
        print(f"===== For coeffMax cutoff at {coeffMax}std =====")

        # ==========  Remove outliers
        print(f"Removing outliers...")
        results = copy(resultsOutliers_dict[tf])
        outlier_coeffMax, coeffMax_thres = kf.get_outliers(
            results['CoeffMax'], coeffMax, 'left',
            METHOD=TRANSFORM_METHOD, PLOT=PLOT
        )
        outliers = results.loc[outlier_coeffMax.index]
        outliers['Topic'] = -1
        results.update(outliers)
        results_dict[tf] = results

        # Convert all topic number into integer results type
        cols = 'Para_id Speech_id Metro Topic WordCount'.split()
        results[cols] = results[cols].astype(int)
        # Compute statistics after outliers are removed
        topicCount_noOutliers = results[results['Topic'] >= 0]
        topicCount_noOutliers = topicCount_noOutliers['Topic'].value_counts().sort_index()
        topicPerc_noOutliers = topicCount_noOutliers/topicCount_noOutliers.sum()
        topics.loc['TopicCount_noOutliers'] = topicCount_noOutliers.sort_values(
            ascending=False)
        topics.loc['Percentage_noOutliers'] = round(
            topicPerc_noOutliers.sort_values(ascending=False)*100, 2)
        print(
            f"{round(len(outliers)/len(results)*100, 2)}% of data removed from removing outliers\n")

        # ========== Construct graphs with varying COEFFMAX
        # ----- Profile actors
        actorProfile, topicMatrix = constructProfile.actors(results_dict[tf])
        # ----- Construct graph
        G, actorProfile, *_ = constructGraph.actorProj(actorProfile, topicMatrix, THRESHOLD)
        # ----- Draw graph
        sns.set_style("dark")
        sns.set_context("talk")
        ax = plt.subplot(pltconfig['nrows'], pltconfig['ncols'], k + 1)
        drawGraph.draw(
            G,
            groupBy=GROUPBY,
            title=f"{TF_LABELS[tf]} | Filter below {coeffMax} STD", title_fontsize=FIG_SIZE*4,
            ax=ax, legend=False,
            node_size=NODE_SIZE,
            edge_width=EGDE_WIDTH,
            node_label=False,
        )

        # ==================================================
        # ----- Plot distribution of coeffMax
        sns.set_style("darkgrid")
        sns.set_context("talk")
        ax = plt.subplot(pltconfig['nrows'], pltconfig['ncols'], (k + 1) + pltconfig['ncols'])
        ax.axvline(x=coeffMax_thres, c=cp.cbPaired['red'])
        sns.distplot(resultsOutliers_dict[tf]['CoeffMax'], kde=False)
        ax.set_xlabel('NMF Coefficient')
        # ==================================================

    plt.tight_layout()
    plt.show()
    # fig.savefig(f"{PATH}plots_varying_coeffMax/ssm_varyingCoeffMax_{nTOPICS}topics_by{GROUPBY}_{tf}.png")
    fig.savefig(f"{PATH}plots_varying_coeffMax/ssm_varyingCoeffMaxwithDist_{nTOPICS}topics_by{GROUPBY}_{tf}.png")

# %% ==========  Remove outliers
resultsOutliers_dict = load_pickle(f"{PATH}ssm_resultsDictOutliers_{nTOPICS}topics_{METHOD}")
topics = pd.read_csv(f"{PATH}ssm_{nTOPICS}topics_{RES}Res_{METHOD}.csv")

print(f"Marking outliers...")
results_dict = {}
for tf in TF_DATES:
    results = copy(resultsOutliers_dict[tf])
    outlier_coeffMax, coeffMax_thres = kf.get_outliers(
        results['CoeffMax'], COEFFMAX[tf],
        'left', METHOD=TRANSFORM_METHOD, PLOT=PLOT
    )
    outliers = results.loc[outlier_coeffMax.index]
    outliers['Topic'] = -1
    results.update(outliers)
    results_dict[tf] = results

    # Convert all topic number into integer results type
    cols = 'Para_id Speech_id Metro Topic WordCount'.split()
    results[cols] = results[cols].astype(int)
    # Compute statistics after outliers are removed
    topicCount_noOutliers = results[results['Topic'] >= 0]
    topicCount_noOutliers = topicCount_noOutliers['Topic'].value_counts().sort_index()
    topicPerc_noOutliers = topicCount_noOutliers/topicCount_noOutliers.sum()
    topics.loc['TopicCount_noOutliers'] = topicCount_noOutliers.sort_values(ascending=False)
    topics.loc['Percentage_noOutliers'] = round(
        topicPerc_noOutliers.sort_values(ascending=False)*100, 2)
    print(topics)
    print(
        f"{round(len(outliers)/len(results)*100, 2)}% of data removed from removing outliers")

    # Save results
    results.to_csv(f"{PATH}ssm_results_{nTOPICS}topics_{METHOD}_{tf}.csv", index=False)
    topics.to_csv(f"{PATH}ssm_{nTOPICS}topics_{METHOD}_{tf}.csv")
save_pickle(results_dict, f"{PATH}ssm_resultsDict_{nTOPICS}topics_{METHOD}")

# ========== Construct actorProfiles and topicMatrix for each time-frame
reload(constructProfile)
results_dict = load_pickle(f"{PATH}ssm_resultsDict_{nTOPICS}topics_{METHOD}")

print("Profiling actors...")
actorProfile_dict = {}
topicMatrix_dict = {}

# ========== Construct attribute profiles for each time-frame
for tf in TF_DATES:
    actorProfile_dict[tf], topicMatrix_dict[tf] = constructProfile.actors(results_dict[tf])

# Save results
save_pickle(results_dict, f"{PATH}ssm_resultsDict_{nTOPICS}topics_{METHOD}")
save_pickle(topicMatrix_dict, f"{PATH}ssm_topicMatrixDict_{nTOPICS}topics_{METHOD}")
print("Actors profiled!")

# ========== Construct Actor & Government Frames
actorFrame_dict = {}
df_cols = 'WordCount SpeechCount'.split() + list(SYM['Topic']['label'].keys()) + ['TotalMentions']
govFrame = pd.DataFrame(columns=df_cols + list(SYM['Topic']['label'].values()))
for tf, actorProfile in actorProfile_dict.items():
    topicList = set(actorProfile.columns).intersection(set(SYM['Topic']['label'].keys()))
    cols = 'WordCount SpeechCount'.split() + list(topicList) + ['TotalMentions']

    # Sort actorProfiles
    actorFrame = actorProfile[cols].sort_values('SpeechCount', ascending=False)
    govFrame_row = actorFrame.sum()
    for topicNum in topicList:
        actorFrame[SYM['Topic']['label'][topicNum]] = actorFrame[topicNum]/actorFrame['TotalMentions']
        govFrame_row[SYM['Topic']['label'][topicNum]] = govFrame_row[topicNum]/govFrame_row['TotalMentions']

    actorFrame = actorFrame.fillna(0)
    actorFrame_dict[tf] = actorFrame
    govFrame.loc[tf] = govFrame_row

    # Export csv for plotting
    actorFrame.to_csv(f"{PATH}csv-export/ssm_actorFrame_{nTOPICS}topics_{tf}.csv")

# Save actor frames
save_pickle(actorFrame_dict, f"{PATH}ssm_actorFrame_{nTOPICS}topics_{METHOD}")
govFrame = govFrame.fillna(0)
govFrame[df_cols] = govFrame[df_cols].astype(int)
govFrame.to_csv(f"{PATH}csv-export/ssm_govFrame_{nTOPICS}topics.csv")

# ========== Construct grouped attribute profiles for each time-frame
groupedProfile_dict = {}
for grp in 'Party PartyAgg Gender Metro'.split():
    groupedProfile_dict[grp] = {}
    for tf in TF_DATES:
        actorProfile = actorProfile_dict[tf]
        groupedProfile = actorProfile.groupby(grp).sum()

        topicList = set(actorProfile.columns).intersection(set(SYM['Topic']['label'].keys()))
        for topicNum in topicList:
            groupedProfile[SYM['Topic']['label'][topicNum]] = groupedProfile[topicNum]/groupedProfile['TotalMentions']

        groupedProfile['ActorCount'] = actorProfile[grp].value_counts()
        cols = groupedProfile.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        groupedProfile = groupedProfile[cols]

        groupedProfile_dict[grp][tf] = groupedProfile
        groupedProfile.to_csv(f"{PATH}csv-export/ssm_groupedFrame_by{grp}_{tf}_{nTOPICS}topics.csv")

# Save results
save_pickle(groupedProfile_dict, f"{PATH}ssm_groupedProfileDict_{nTOPICS}topics_{METHOD}")

# ========== Construct Actor-projection Graphs for all time-frames
reload(constructGraph)

apGraph_dict = {}
apConnectedComp = {}
for tf in TF_DATES:
    print(f"Constructing {tf} actor-projection...")
    # ----- Construct graph
    apGraph_dict[tf], actorProfile_dict[tf], apConnectedComp[tf] = constructGraph.actorProj(
        actorProfile_dict[tf], topicMatrix_dict[tf],
        THRESHOLD)

# ----- Save all results
save_pickle(apGraph_dict, f"{PATH}/ssm_graphALL_{nTOPICS}topics_{METHOD}")
save_pickle(actorProfile_dict, f"{PATH}ssm_actorProfileDict_{nTOPICS}topics_{METHOD}")
print("Actor-projection graph constructed!")

 #%% ========== Generate Graph drawing data
reload(drawGraph)

def byOrd4(data):
    if 0 < data < 0.5:
        return 0
    elif 0.5 <= data < 0.75:
        return 1
    elif 0.75 <= data < 0.875:
        return 2
    elif data >= 0.875:
        return 3
    else:
        return np.nan


def byThres(data, thres):
    if data >= thres:
        return 1
    else:
        return 0


def byName(data, name):
    if data == name:
        return 1
    else:
        return 0

# Load data
actorProfile_dict = load_pickle(f"{PATH}ssm_actorProfileDict_{nTOPICS}topics_{METHOD}")
apGraph_dict = load_pickle(f"{PATH}/ssm_graphALL_{nTOPICS}topics_{METHOD}")

nodeAttr_dict = {}
edgeAttr_dict = {}
actor_highCent = 'Tanya Plibersek'
for tf in TF_DATES:
    G = apGraph_dict[tf]
    AP = actorProfile_dict[tf]

    # ----- Features by actor's attribute
    nodeAttr_dict[tf] = pd.DataFrame(index=AP.index)
    nodeAttr_dict[tf]['label'] = [name.replace(' ', '\n') for name in AP.index]
    for grp in 'Party PartyAgg Gender Metro'.split():
        attr_series = AP[grp].apply(lambda x: SYM[grp]['colour'][x])
        nodeAttr_dict[tf] = nodeAttr_dict[tf].join(attr_series)

    # ----- Special features for nodes with high centrality
    # Interval attributes for nodes requires scaling
    thres_highCent = AP['BtwnCentrality'].max()
    for attr in 'node_size  node_edgewidth font_size'.split():
        # attr_series = AP['BtwnCentrality'].apply(
        #     lambda x: EMP_MAP[attr][byThres(x, thres_highCent)])

        AP_names = pd.Series(AP.index, index=AP.index)
        attr_series = AP_names.apply(lambda x: EMP_MAP[attr][byName(x, actor_highCent)])

        nodeAttr_dict[tf] = nodeAttr_dict[tf].join(attr_series.rename(attr)*G_STYLE[attr])
    # Nominal attributes for nodes does not require scaling
    for attr in 'node_alpha node_edgecolour font_colour font_weight font_alpha'.split():
        # attr_series = AP['BtwnCentrality'].apply(
        #     lambda x: EMP_MAP[attr][byThres(x, thres_highCent)])

        AP_names = pd.Series(AP.index, index=AP.index)
        attr_series = AP_names.apply(lambda x: EMP_MAP[attr][byName(x, actor_highCent)])

        nodeAttr_dict[tf] = nodeAttr_dict[tf].join(attr_series.rename(attr))

    # ----- Features of edges
    edges = pd.Series(nx.get_edge_attributes(G, 'weight'))
    edgeAttr_dict[tf] = pd.DataFrame(index=G.edges)
    # Interval attributes for edges requires scaling
    edge_series = edges.apply(lambda x: EMP_MAP['edge_width'][byOrd4(x)])
    edgeAttr_dict[tf] = edgeAttr_dict[tf].join(edge_series.rename('edge_width'))
    edgeAttr_dict[tf]['edge_width'] = edgeAttr_dict[tf]['edge_width']*G_STYLE['edge_width']
    # Nominal attributes for edges does not require scaling
    edge_series = edges.apply(lambda x: EMP_MAP['edge_colour'][byOrd4(x)])
    edgeAttr_dict[tf] = edgeAttr_dict[tf].join(edge_series.rename('edge_colour'))

    # ----- Special features for edges of node with high centrality
    # actor_highCent = AP.index[AP['BtwnCentrality'] == thres_highCent][0]
    edge_series = copy(edges)
    # Edges are stored with multi-indices. Use IndexSlice to retrieve elements!
    idx = pd.IndexSlice
    edge_series.loc[:] = EMP_MAP['edge_alpha'][0]
    try:
        edge_series.loc[idx[actor_highCent, :]] = EMP_MAP['edge_alpha'][1]
    except:
        pass
    try:
        edge_series.loc[idx[:, actor_highCent]] = EMP_MAP['edge_alpha'][1]
    except:
        pass
    edgeAttr_dict[tf] = edgeAttr_dict[tf].join(edge_series.rename('edge_alpha'))


# %% ========== Draw Actor-projection Graphs (no subplots)
reload(drawGraph)
timeframes = TIME_FRAMES['termsPM']
tf = 'Turnbull-PostSurvey'
grp = 'Party'
# Load data
apGraph_dict = load_pickle(f"{PATH}/ssm_graphALL_{nTOPICS}topics_{METHOD}")

sns.set_style("dark")
sns.set_context("talk")
FIG_SIZE = 5

for tf in timeframes:
    fig = plt.figure(figsize=(FIG_SIZE*1.5, FIG_SIZE), dpi=300)
    pos = nx.kamada_kawai_layout(apGraph_dict[tf], weight='weight')
    # ----- Draw elements that are non-high centrality
    # nx.draw_networkx(
    #     apGraph_dict[tf], pos=pos,
    #     node_color=nodeAttr_dict[tf][grp],
    #     node_size=G_STYLE['node_size'],
    #     linewidths=G_STYLE['node_edgewidth'],
    #     edgecolors=G_STYLE['node_edgecolour'],
    #     alpha=G_STYLE['node_alpha'],
    #
    #     labels=nodeAttr_dict[tf]['label'],
    #     font_size=G_STYLE['font_size'],
    #     font_weight=G_STYLE['font_weight'],
    #     font_color=G_STYLE['font_colour'],
    #
    #     width=edgeAttr_dict[tf]['edge_width'],
    #     edge_color=edgeAttr_dict[tf]['edge_colour'],
    # )
    drawGraph.draw(
        apGraph_dict[tf],
        groupBy=grp.lower(),
        node_size=G_STYLE['node_size'],
        edge_width=G_STYLE['edge_width'],
        node_edgewidth=G_STYLE['node_edgewidth'],
        font_size=G_STYLE['font_size'],
        legend=False,
    )

    plt.title(f"{TF_LABELS[tf]} | Filter below {COEFFMAX[tf]} STD", size=FIG_SIZE*2)
    plt.tight_layout()
    fig.show()
    fig.savefig(f"{PATH}/by{grp}/ssm_apGraph_{tf}_{nTOPICS}topics.png")

# %% ========== Draw Actor-projection Graphs (high cent emphasis, no subplots)
reload(drawGraph)
timeframes = TIME_FRAMES['termsPM']
tf = 'Turnbull-PostSurvey'
grp = 'Party'
# Load data
apGraph_dict = load_pickle(f"{PATH}/ssm_graphALL_{nTOPICS}topics_{METHOD}")

sns.set_style("dark")
sns.set_context("talk")
FIG_SIZE = 5

# for grp in 'Party PartyAgg Gender Metro'.split():
for tf in timeframes:
    AP = actorProfile_dict[tf]
    # actor_highCent = AP.index[AP['BtwnCentrality'] == AP['BtwnCentrality'].max()][0]
    actor_highCent = 'Tanya Plibersek'

    nodeAttr = nodeAttr_dict[tf]
    highCent_node = nodeAttr.loc[actor_highCent]
    nonHighCent_node = nodeAttr[~(nodeAttr.index == actor_highCent)]

    edgeAttr = edgeAttr_dict[tf]
    highCent_edge = pd.concat(
        [edgeAttr.loc[idx[:, actor_highCent], :], edgeAttr.loc[idx[actor_highCent, :], :]])
    nonHighCent_edge = edgeAttr.loc[edgeAttr.index.difference(highCent_edge.index)]

    fig = plt.figure(figsize=(FIG_SIZE*1.5, FIG_SIZE), dpi=300)
    pos = nx.kamada_kawai_layout(apGraph_dict[tf], weight=None)
    # ----- Draw elements that are non-high centrality
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=nonHighCent_node.index.tolist(),
        node_size=nonHighCent_node['node_size'],
        node_color=nonHighCent_node[grp],
        alpha=nonHighCent_node['node_alpha'],
        linewidths=nonHighCent_node['node_edgewidth'],
        edgecolors=nonHighCent_node['node_edgecolour'],
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=nonHighCent_edge.index.tolist(),
        width=nonHighCent_edge['edge_width'],
        edge_color=nonHighCent_edge['edge_colour'],
        alpha=EMP_MAP['edge_alpha'][0],
    )
    nx.draw_networkx_labels(
        G, pos,
        labels=nonHighCent_node['label'],
        font_size=EMP_MAP['font_size'][0]*G_STYLE['font_size'],
        font_color=EMP_MAP['font_colour'][0],
        font_weight=EMP_MAP['font_weight'][0],
        alpha=EMP_MAP['font_alpha'][0],
    )
    # ----- Draw elements that possesses high centrality
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[actor_highCent],
        node_size=highCent_node['node_size'],
        node_color=highCent_node[grp],
        alpha=highCent_node['node_alpha'],
        linewidths=highCent_node['node_edgewidth'],
        edgecolors=highCent_node['node_edgecolour'],
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=highCent_edge.index.tolist(),
        width=highCent_edge['edge_width'],
        edge_color='black',
        alpha=EMP_MAP['edge_alpha'][1],
    )
    nx.draw_networkx_labels(
        G, pos,
        labels={actor_highCent: highCent_node['label']},
        font_size=EMP_MAP['font_size'][1]*G_STYLE['font_size'],
        font_color=EMP_MAP['font_colour'][1],
        font_weight=EMP_MAP['font_weight'][1],
        alpha=EMP_MAP['font_alpha'][1],
    )

    plt.title(f"{TF_LABELS[tf]} | Filter below {COEFFMAX[tf]} STD", size=FIG_SIZE*2)
    plt.tight_layout()
    fig.show()
    fig.savefig(f"{PATH}/by{grp}/ssm_apGraphEmp_{tf}_{nTOPICS}topics.png")

# %% ========== Draw Actor-projection Graphs (subplots)
reload(drawGraph)
# Load data
apGraph_dict = load_pickle(f"{PATH}/ssm_graphALL_{nTOPICS}topics_{METHOD}")

sns.set_style("dark")
sns.set_context("talk")
FIG_SIZE = 5

for tfLabel, timeframes in TIME_FRAMES.items():
    for grp in 'Party PartyAgg Gender Metro'.split():
        pltconfig = {'nrows': np.ceil(len(timeframes)/2), 'ncols': 2}
        fig = plt.figure(figsize=(FIG_SIZE*1.5*pltconfig['ncols'], FIG_SIZE*pltconfig['nrows']),
                         dpi=300)

        for i, tf in enumerate(timeframes):
            # ----- Draw graph
            ax = plt.subplot(pltconfig['nrows'], pltconfig['ncols'], i + 1)
            pos = nx.kamada_kawai_layout(apGraph_dict[tf], weight=None)
            nx.draw_networkx(
                apGraph_dict[tf], pos=pos,
                node_color=nodeAttr_dict[tf][grp],
                node_size=G_STYLE['node_size'],
                linewidths=G_STYLE['node_edgewidth'],
                edgecolors=G_STYLE['node_edgecolour'],
                alpha=G_STYLE['node_alpha'],

                labels=nodeAttr_dict[tf]['label'],
                font_size=G_STYLE['font_size'],
                font_weight=G_STYLE['font_weight'],
                font_color=G_STYLE['font_colour'],

                width=edgeAttr_dict[tf]['edge_width'],
                edge_color=edgeAttr_dict[tf]['edge_colour'],
            )
            plt.title(f"{TF_LABELS[tf]} | Filter below {COEFFMAX[tf]} STD", size=FIG_SIZE*2)
        plt.tight_layout()
        fig.show()
        fig.savefig(f"{PATH}ssm_apGraph_{tfLabel}_{nTOPICS}topics_by{grp}.png")

# %% ========== Line-plot of percentage of topic mentioned in all time-frames
timeframes = TIME_FRAMES['termsPM']
# Load data
actorProfile_dict = load_pickle(f"{PATH}ssm_actorProfileDict_{nTOPICS}topics_{METHOD}")

topicMentions = pd.DataFrame(columns=SYM['Topic']['label'].keys())
for tf in timeframes:
    cols = set(SYM['Topic']['label'].keys()).intersection(actorProfile_dict[tf].columns)
    oneTerm = actorProfile_dict[tf][cols]
    topicMentions.loc[tf] = oneTerm.sum()/oneTerm.sum().sum()
topicMentions = topicMentions.fillna(0)
topicMentions.to_csv(f"{PATH}ssm_topicMentions_{RES}Res_{nTOPICS}topics.csv")

FIG_SIZE = 4
sns.set_style("darkgrid")
sns.set_context("notebook")
pltconfig = {'nrows': 2, 'ncols': 2}
fig = plt.figure(figsize=(FIG_SIZE*1.5*pltconfig['ncols'], FIG_SIZE*pltconfig['nrows']),
                 dpi=300)
for topic in topicMentions.columns:
    sns.lineplot(
        timeframes, topicMentions[topic],
        color=SYM['Topic']['colourLight'][topic], linewidth=FIG_SIZE,
    )
    sns.scatterplot(
        timeframes, topicMentions[topic],
        color=SYM['Topic']['colourLight'][topic],
        s=FIG_SIZE*100, marker=SYM['Topic']['marker'][topic],
    )
xtickslabels = list(map(lambda x: x.replace(', ', '\n'), TF_LABELS.loc[timeframes].values))
plt.xticks(timeframes, xtickslabels)
plt.ylabel("Percentage")
plt.title(f"Topic Mentions", fontsize=FIG_SIZE*3)
plt.legend(SYM['Topic']['label'].values(), fontsize=FIG_SIZE*3)

fig.tight_layout()
fig.show()
fig.savefig(f"{PATH}ssm_topicMentions_{RES}Res_{nTOPICS}topics.png")

# %% ========== Lineplot of percentage of topic mentioned of in all time-frames grouped by party
timeframes = TIME_FRAMES['termsPM']
# Load data
groupedProfile_dict = load_pickle(f"{PATH}ssm_groupedProfileDict_{nTOPICS}topics_{METHOD}")

topicMentionsGrouped_dict = {}
for grp in 'Party PartyAgg Gender Metro'.split():
    topicMentionsGrouped_dict[grp] = {}
    for attr in SYM[grp]['label']:
        topicMentionsGrouped_dict[grp][attr] = pd.DataFrame(
            columns=SYM['Topic']['label'].keys())
        for tf in timeframes:
            # Only use columns in groupedProfile
            cols = set(SYM['Topic']['label'].keys()).intersection(
                groupedProfile_dict[grp][tf].columns)
            if attr in groupedProfile_dict[grp][tf].index:
                # Retrieve topicMentions in one term and normalise
                oneTerm = groupedProfile_dict[grp][tf][cols].loc[attr]
                topicMentionsGrouped_dict[grp][attr].loc[tf] = oneTerm/oneTerm.sum()
            else:
                topicMentionsGrouped_dict[grp][attr].loc[tf] = np.nan
        topicMentionsGrouped_dict[grp][attr] = topicMentionsGrouped_dict[grp][attr].fillna(0)
        topicMentionsGrouped_dict[grp][attr].to_csv(
            f"{PATH}ssm_topicMentions_by{grp}{attr}_{RES}Res_{nTOPICS}topics.csv")
save_pickle(topicMentionsGrouped_dict,
            f"{PATH}ssm_topicMentionsGroupedDict_{RES}Res_{nTOPICS}topics.png")

sns.set_style("darkgrid")
sns.set_context("notebook")
FIG_SIZE = 5
for grp in 'PartyAgg Gender Metro'.split():
    pltconfig = {'nrows': np.ceil(len(topicMentionsGrouped_dict[grp])/2), 'ncols': 2}
    fig = plt.figure(figsize=(FIG_SIZE*1.5*pltconfig['ncols'], FIG_SIZE*pltconfig['nrows']),
                     dpi=300)
    for k, (attr, attrLabel) in enumerate(SYM[grp]['label'].items()):
        plt.subplot(pltconfig['nrows'], pltconfig['ncols'], k + 1)
        for topic in topicMentionsGrouped_dict[grp][attr].columns:
            sns.lineplot(
                timeframes, topicMentionsGrouped_dict[grp][attr][topic],
                color=SYM['Topic']['colourLight'][topic], linewidth=FIG_SIZE*0.75,
            )
            sns.scatterplot(
                timeframes, topicMentionsGrouped_dict[grp][attr][topic],
                color=SYM['Topic']['colourLight'][topic], s=FIG_SIZE*50,
                marker=SYM['Topic']['marker'][topic],
            )
        plt.ylabel("Percentage")
        plt.title(f"Topics Mentions | {attrLabel}", fontsize=FIG_SIZE*3)
        if k == 0:
            plt.legend(SYM['Topic']['label'].values(), fontsize=FIG_SIZE*2)

    fig.tight_layout()
    fig.show()
    fig.savefig(f"{PATH}ssm_topicMentions_{RES}Res_{nTOPICS}topics_by{grp}.png")

# %% ========== Plot Correlation Heatmap
timeframes = TIME_FRAMES['termsPM']
actorProfile_dict = load_pickle(f"{PATH}ssm_actorProfileDict_{nTOPICS}topics_{METHOD}")

sns.set_style("darkgrid")
sns.set_context("notebook")
FIG_SIZE = 5
pltconfig = {'nrows': np.ceil(len(actorProfile_dict)/2), 'ncols': 2}
fig = plt.figure(figsize=(FIG_SIZE*1.5*pltconfig['ncols'], FIG_SIZE*pltconfig['nrows']))

for k, tf in enumerate(timeframes):
    topicList = set(actorProfile_dict[tf].columns).intersection(set(SYM['Topic']['label'].keys()))
    xtickLabels = [SYM['Topic']['label'][topic].replace(' ', '\n') for topic in topicList]
    ytickLabels = [SYM['Topic']['label'][topic] for topic in topicList]

    ax = plt.subplot(pltconfig['nrows'], pltconfig['ncols'], k + 1)
    sns.heatmap(
        actorProfile_dict[tf][topicList].corr(), annot=True, ax=ax,
        xticklabels=xtickLabels, yticklabels=ytickLabels,
        cmap='coolwarm', center=0,
        linewidths=3,
    )
    ax.set_title(f"{TF_LABELS[tf]}")
plt.tight_layout()
plt.savefig(f"{PATH}ssm_topicCorrelation_{RES}Res_{nTOPICS}topics.png")
plt.show()

# %% ========== Plot Correlation Heatmap
actorProfile_dict = load_pickle(f"{PATH}ssm_actorProfileDict_{nTOPICS}topics_{METHOD}")
tf = '45thPostPleb'

sns.set_style("darkgrid")
sns.set_context("notebook")
FIG_SIZE = 5
fig = plt.figure(figsize=(FIG_SIZE*1.5, FIG_SIZE), dpi=300)

topicList = set(actorProfile_dict[tf].columns).intersection(set(SYM['Topic']['label'].keys()))
xtickLabels = [SYM['Topic']['label'][topic].replace(' ', '\n') for topic in topicList]
ytickLabels = [SYM['Topic']['label'][topic] for topic in topicList]

sns.heatmap(
    actorProfile_dict[tf][topicList].corr(), annot=True,
    xticklabels=xtickLabels, yticklabels=ytickLabels,
    cmap='coolwarm', center=0,
    linewidths=3,
)
plt.title(f"{TF_LABELS[tf]}")
plt.tight_layout()
plt.savefig(f"{PATH}ssm_topicCorrelation_{RES}Res_{nTOPICS}topics_{tf}.png")
plt.show()

#%% Pearson's Correlation Test
actorProfile_dict = load_pickle(f"{PATH}ssm_actorProfileDict_{nTOPICS}topics_{METHOD}")

topicCorr_dict = {}
for tf in TIME_FRAMES['termsPM']:
    actorProfile = actorProfile_dict[tf]
    topicList = set(actorProfile.columns).intersection(set(SYM['Topic']['label'].keys()))
    topicCorr = []
    for topic1, topic2 in iter.combinations(topicList, 2):
        topicCorr.append(stats.pearsonr(actorProfile[topic1], actorProfile[topic2]))

    topicCorr_dict[tf] = pd.DataFrame(
        topicCorr,
        index=pd.MultiIndex.from_tuples(list(iter.combinations(topicList, 2))),
        columns='r p-value'.split()
    )
    print(tf, topicCorr_dict[tf][topicCorr_dict[tf]['p-value'] < 0.05])

# %% ========== Plot Top-n Betweenness Centrality
timeframes = TIME_FRAMES['termsPM']
# Load data
actorProfile_dict = load_pickle(f"{PATH}ssm_actorProfileDict_{nTOPICS}topics_{METHOD}")

FIG_SIZE = 5
nTOP = 5
apCentBetweenness = pd.DataFrame(columns='1st 2nd 3rd 4th 5th'.split())
for tf in timeframes:
    apCentBetweenness.loc[tf] = actorProfile_dict[tf]['BtwnCentrality'].sort_values(ascending=False)[
                                :nTOP].tolist()
apCentBetweenness.to_csv(f"{PATH}ssm_centBetweennessTop{nTOP}_{RES}Res_{nTOPICS}topics.csv")

ax = apCentBetweenness.plot.bar(
    rot=0, cmap='summer',
    title=f"Top{nTOP} Betweenness Centrality",
    figsize=(FIG_SIZE*1.5, FIG_SIZE),
)
fig = ax.get_figure()
fig.set_dpi(300)

fig.tight_layout()
fig.show()
fig.savefig(f"{PATH}ssm_centBetweennessTop{nTOP}_{RES}Res_{nTOPICS}topics.png")

# %% ========== Plot centrality distribution for all time-frame
timeframes = TIME_FRAMES['termsPM']
# Load data
actorProfile_dict = load_pickle(f"{PATH}ssm_actorProfileDict_{nTOPICS}topics_{METHOD}")

FIG_SIZE = 5
sns.set_style("darkgrid")
sns.set_context("notebook")
pltconfig = {'nrows': np.ceil(len(timeframes)/2), 'ncols': 2}

for cent_type in 'DegreeCentrality BtwnCentrality'.split():
    fig = plt.figure(figsize=(FIG_SIZE*1.5*pltconfig['ncols'], FIG_SIZE*pltconfig['nrows']),
                     dpi=300)
    for i, tf in enumerate(timeframes):
        ax = plt.subplot(pltconfig['nrows'], pltconfig['ncols'], i + 1)
        sns.distplot(actorProfile_dict[tf][cent_type], kde=False, rug=True, color=cp.cbPaired['red'])
        ax.set_title(f"{TF_LABELS[tf]}: {cent_type} Centrality")

        if cent_type == 'DegreeCentrality':
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 60)
        elif cent_type == 'BtwnCentrality':
            ax.set_xlim(0, 0.04)
            ax.set_ylim(0, 70)

    fig.tight_layout()
    fig.show()
    fig.savefig(f"{PATH}ssm_cent{cent_type}_{RES}Res_{nTOPICS}topics.png")

# %%# ========== Draw all Dendrograms and determine threshold to split cliques
timeframes = TIME_FRAMES['termsPM']
DENDRO_THRES = {
    'All'         : 0.1,
    # '40th'        : 0.1,
    # '43rd'        : 0.1,
    # '44th'        : 0.1,
    # '45th'        : 0.1,
    # '45thPrePleb' : 0.1,
    # '45thPostPleb': 0.1,
    'Howard'  : 0.1,
    'Gillard' : 0.1,
    'Abbott'  : 0.1,
    'Turnbull': 0.1,
    'Turnbull-PreSurvey': 0.1,
    'Turnbull-PostSurvey': 0.2,
}
# Load data
topicMatrix_dict = load_pickle(f"{PATH}ssm_topicMatrixDict_{nTOPICS}topics_{METHOD}")

# Remove rows with all zeroes
topicMatrixNoZeros_dict = copy(topicMatrix_dict)
for tf in TF_DATES:
    for actor, row in topicMatrixNoZeros_dict[tf].iterrows():
        if row.tolist() == [0]*len(topicMatrixNoZeros_dict[tf].columns):
            topicMatrixNoZeros_dict[tf] = topicMatrixNoZeros_dict[tf].drop(actor)

# ----- Evaluated which method of clustering is best using Cophenetic Correlation
coph_allMethods = pd.DataFrame(index=TF_DATES)
# Methods appropriate for cosine similarity metric are 'single', 'average', 'weighted'
for method in 'single average weighted'.split():
    cophList = []
    for tf in TF_DATES:
        # 'linked' is the array that shows how each actor is clustered together
        linked = linkage(topicMatrixNoZeros_dict[tf].to_numpy(), method=method,
                         metric='cosine')
        # Compute cophenetic correlation
        coph, _ = cophenet(linked, pdist(topicMatrixNoZeros_dict[tf].to_numpy()))
        cophList.append(coph)
    coph_allMethods[method] = cophList
# Compute average cophenetic correlation for each method
coph_allMethods.loc['Mean'] = coph_allMethods.mean()
# Retrieve best method
bestMethod = \
coph_allMethods.columns[coph_allMethods.loc['Mean'] == coph_allMethods.loc['Mean'].max()][0]
print(
    f"Best Method is '{bestMethod.capitalize()}' with coeff={round(coph_allMethods.loc['Mean'].max(), 2)}")

# ----- Draw Dendrograms
FIG_SIZE = 5
plt.style.use('seaborn')
pltconfig = {'nrows': np.ceil(len(timeframes)/3), 'ncols': 3}
fig = plt.figure(figsize=(FIG_SIZE*pltconfig['ncols'], FIG_SIZE*pltconfig['nrows']*2),
                 dpi=300)
linked_dict = {}

for i, tf in enumerate(timeframes):
    print(f"Drawing {tf} Dendrogram...")
    # 'linked_dict' is the array used to draw a dendrogram
    linked_dict[tf] = linkage(topicMatrixNoZeros_dict[tf].to_numpy(), method=bestMethod,
                              metric='cosine')
    # Retrieve last name of actors
    nameList = list(
        map(lambda x: re.sub('.* ', '', x), list(topicMatrixNoZeros_dict[tf].index)))

    ax = plt.subplot(pltconfig['nrows'], pltconfig['ncols'], i + 1)
    dendrogram(
        linked_dict[tf],
        orientation='right', leaf_font_size=FIG_SIZE*2, labels=nameList,
        color_threshold=DENDRO_THRES[tf],
        # above_threshold_color='gray',
        truncate_mode='lastp', p=40, show_contracted=True,
    )
    plt.axvline(x=DENDRO_THRES[tf], color='b', linestyle='--')
    ax.set_title(f"{TF_LABELS[tf]}", fontsize=FIG_SIZE*3)
    ax.tick_params(axis='x', which='major', labelsize=FIG_SIZE*2)
    ax.set_xlabel("Cosine Similarity", fontsize=FIG_SIZE*2)

fig.tight_layout()
fig.show()
fig.savefig(f"{PATH}ssm_dendrograms_{RES}Res_{nTOPICS}topics.png")
print("Dendrograms drawn!")


# %% ========== Find cliques for all time-frames
def findCliques(topicMatrixNoZeros, actorProfile, THRES, method='single',
                criterion='inconsistent'):
    # Retrieve clusters
    linked = linkage(topicMatrixNoZeros.to_numpy(), method=method, metric='cosine')
    cluster = fcluster(linked, THRES, criterion=criterion)

    # Construct clique profile
    cliqueProfile = pd.DataFrame(columns=actorProfile.columns, index=topicMatrixNoZeros.index)
    for actor, row in actorProfile.iterrows():
        if actor in cliqueProfile.index:
            cliqueProfile.loc[actor] = actorProfile.loc[actor]
    cliqueProfile['Clique'] = cluster
    cliqueProfile = cliqueProfile.sort_values('Clique')
    return cliqueProfile


# Load data
actorProfile_dict = load_pickle(f"{PATH}ssm_actorProfileDict_{nTOPICS}topics_{METHOD}")
# Construct clique profiles
cliqueProfile_dict = {}
for tf in TF_DATES:
    cliqueProfile_dict[tf] = findCliques(topicMatrixNoZeros_dict[tf], actorProfile_dict[tf],
                                         DENDRO_THRES[tf], method=bestMethod,
                                         criterion='distance')

# ----- Generate and plot topic profiles for each clique in each time-frame
reload(drawGraph)
timeframes = TIME_FRAMES['termsPM']

cliqueAttr_dict = {}
for tf in TF_DATES:
    # Construct table of normalised attribute count
    cliqueAttr = pd.DataFrame(columns=cliqueProfile_dict[tf]['Clique'].unique(),
                              index=CLQ_PROFILE_COLOURS.keys())
    for clq in cliqueProfile_dict[tf]['Clique'].unique():
        # Retrieve rows of one clique
        oneClique = cliqueProfile_dict[tf][cliqueProfile_dict[tf]['Clique'] == clq]

        # ----- Compute normalised clique size
        cliqueAttr_col = pd.Series()
        cliqueAttr_col.loc['Size'] = len(oneClique)/len(cliqueProfile_dict[tf])

        # ----- Compute normalised count for actor attributes
        for grp in 'Party Gender Metro'.split():
            # Count attribute and normalise
            oneAttr = oneClique.groupby(grp).count()['Clique']
            oneAttr = oneAttr/oneAttr.sum()
            if grp != 'Party':
                oneAttr.index = [SYM[grp]['label'][index] for index in oneAttr.index]
            cliqueAttr_col = cliqueAttr_col.append(oneAttr)

        # ----- Compute normalised count for topics
        # Parse topicList
        cols = set(SYM['Topic']['label'].keys()).intersection(oneClique.columns)
        topicMentions = oneClique[cols].sum()/oneClique[cols].sum().sum()
        topicMentions.index = [SYM['Topic']['label'][index] for index in topicMentions.index]
        cliqueAttr_col = cliqueAttr_col.append(topicMentions)

        cliqueAttr[clq] = cliqueAttr_col
    cliqueAttr_dict[tf] = cliqueAttr

# Plot cliques bar graphs
FIG_SIZE = 5
sns.set_style("darkgrid")
sns.set_context("notebook")
for tf in timeframes:
    print(f"Plotting {tf} Cliques...")
    pltconfig = {'nrows': np.ceil(len(cliqueAttr_dict[tf].columns)/2), 'ncols': 2}
    fig = plt.figure(figsize=(FIG_SIZE*3*pltconfig['ncols'], FIG_SIZE*pltconfig['nrows']))
    for k, clq in enumerate(cliqueAttr_dict[tf].columns):
        ax = plt.subplot(pltconfig['nrows'], pltconfig['ncols'], k + 1)
        sns.barplot(
            cliqueAttr_dict[tf].index, cliqueAttr_dict[tf][clq],
            palette=CLQ_PROFILE_COLOURS, edgecolor='black', ax=ax,
        )
        ax.set_title(
            f"{tf} | Clique {clq} | Actor Count:{int(cliqueAttr_dict[tf][clq]['Size']*len(cliqueProfile_dict[tf]))}",
            fontsize=FIG_SIZE*3,
        )
        ax.set_ylim((0, 1.1))
        ax.set_ylabel('')
        xticklabels = []
        for label in cliqueAttr_dict[tf].index:
            if type(label) == str:
                xticklabels.append(label.replace(' ', '\n'))
            else:
                xticklabels.append(label)
        ax.set_xticklabels(xticklabels)

    fig.tight_layout()
    fig.show()
    fig.savefig(f"{PATH}ssm_cliqueProfiles_{RES}Res_{nTOPICS}topics_{tf}.png")
print("Clique Profiles Plotted!")


