"""
@author: kaisoon
"""
# ----- Imports and Functions
import pandas as pd
from datetime import datetime
import importlib
import miscFuncs

#%% Clean Data
importlib.reload(miscFuncs)

FILE_NAME = "marriage_bills_para_short_cleaned.csv"
data = pd.read_csv(f"data/raw/{FILE_NAME}")

data_cleaned = pd.DataFrame(columns='Speech_id Date Bill Type Person Gender Party Elec Metro Speech'.split())
speechIds = data['speech_id'].unique()

# ----- Concatenate all paragraphs of the same speech into one
for sID in speechIds:
    # Extract rows that are of the same speech
    sameSpeech = data[data['speech_id'] == sID]
    # Concat all paragraphs of the speech into one
    fullSpeech = ""
    for p in sameSpeech['para']:
        fullSpeech = fullSpeech + p

    # ----- Replace all "per cent" in speeches with "percent"
    fullSpeech = fullSpeech.replace("per cent", "percent")
    # ----- Replace all "'ve" in speeches with "have"
    fullSpeech = fullSpeech.replace("'ve ", " have ")
    # ----- Replace all "'ll" in speeches with "will"
    fullSpeech = fullSpeech.replace("'ll ", " will ")

    # Assemble all data for a row
    row = [
        sameSpeech.iloc[0]['speech_id'],
        sameSpeech.iloc[0]['Date'].strip(),
        sameSpeech.iloc[0]['Bill'].strip(),
        sameSpeech.iloc[0]['Type'].strip(),
        miscFuncs.rearrangeName(sameSpeech.iloc[0]['Person'].strip()),
        sameSpeech.iloc[0]['Gender'],
        sameSpeech.iloc[0]['Party'].strip(),
        sameSpeech.iloc[0]['Elec'].strip(),
        sameSpeech.iloc[0]['metro'],
        fullSpeech
    ]
    row = pd.Series(row, index=data_cleaned.columns)
    # Append row to data_cleaned
    data_cleaned = data_cleaned.append(row, ignore_index=True)

# Sort and save data
data_cleaned.sort_values(by=['Date', 'Bill'], inplace=True)
data_cleaned.to_csv("data/ssm_cleaned.csv", index=False)


#%% ----- Remove Irrelevant Data
# Remove speeches that are first readings
data_rel = data_cleaned[data_cleaned['Type'] != 'First Reading']

# Remove speeches associated with superannuation
index = [i for i,row in data_rel.iterrows() if 'superannuation' not in row['Bill'].lower()]
data_rel = data_rel.loc[index]


# Sort and save data
data_rel.sort_values(by=['Date', 'Bill'], inplace=True)
data_rel.to_csv("data/ssm_rel.csv", index=False)


#%% Divide results into time-frames
importlib.reload(miscFuncs)

data = pd.read_csv("data/ssm_rel.csv")
results = pd.read_csv("results/ssm_results_nmf_senti.csv")
results = miscFuncs.convert2datetime(results)

# ---------------------------------------------------------------------------
# 2004: Bills from 2004-05-27 to 2004-06-24
# Howard amended Marriage Act 1961 to be defined as a "union of a man and a woman to the exclusion of all others"
# ---------------------------------------------------------------------------
# 2012: Bills from 2012-02-13 to 2012-09-10
# First time House of Reps declared their position on SSM
# ---------------------------------------------------------------------------
# 2013: Bills from 2013-03-18 to 2013-06-24
# New Zealand legislated for SSM on the 17th April
# ---------------------------------------------------------------------------
# 2015: Bills from 2015-06-01 to 2015-11-23
# US legalised nation-wide SSM on the 26th June
# ---------------------------------------------------------------------------
# 2016: Bills from 2016-02-08 to 2016-11-21
# SSM plebiscite done between 12th Sept and 7th Nov
# ---------------------------------------------------------------------------
# 2017: Bills from 2017-09-13 to 2017-12-07
# Plebiscite results released on 15th Nov. SSM law passed on 7th Dec
# ---------------------------------------------------------------------------
resultsDict = {}
resultsDict['2004'] = results[(results['Date'] >= datetime(2004, 5, 27)) & (results['Date'] <= datetime(2004, 6, 24))]
resultsDict['2012'] = results[(results['Date'] >= datetime(2012, 2, 13)) & (results['Date'] <= datetime(2012, 9, 10))]
resultsDict['2013'] = results[(results['Date'] >= datetime(2013, 3, 18)) & (results['Date'] <= datetime(2013, 6, 24))]
resultsDict['2015'] = results[(results['Date'] >= datetime(2015, 6, 1)) & (results['Date'] <= datetime(2015, 11, 23))]
resultsDict['2016'] = results[(results['Date'] >= datetime(2016, 2, 8)) & (results['Date'] <= datetime(2017, 11, 21))]
resultsDict['2017'] = results[(results['Date'] >= datetime(2017, 9, 13)) & (results['Date'] <= datetime(2017, 12, 7))]

for k,v in resultsDict.items():
    print(f"{k:{10}}{len(v)} speeches")
    resultsDict[k].to_csv(f"results/{k}/ssm_results_{k}.csv", index=False)
