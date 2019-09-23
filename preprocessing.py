"""
@author: kaisoon
"""
# ----- Imports and Functions
import os
import time as tm
import pandas as pd, numpy as np
import concurrent.futures
import importlib
import miscFuncs
import spacy
nlp = spacy.load('en_core_web_sm')


#%% Clean Data
importlib.reload(miscFuncs)
startTime = tm.perf_counter()

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
    # ----- Replace all "'same sex" in speeches with "same-sex"
    fullSpeech = fullSpeech.replace("same sex", "same-sex")

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

dur = tm.gmtime(tm.perf_counter() - startTime)
print(f"Cleaning completed in {dur.tm_sec}s")


#%% ----- Remove Irrelevant Data
# Remove speeches that are first readings
data_rel = data_cleaned[data_cleaned['Type'] != 'First Reading']

# Remove speeches associated with superannuation
index = [i for i,row in data_rel.iterrows() if 'superannuation' not in row['Bill'].lower()]
data_rel = data_rel.loc[index]


# Sort and save data
data_rel.sort_values(by=['Date', 'Bill'], inplace=True)
data_rel.to_csv("data/ssm_rel.csv", index=False)


#%% ----- Lemmatise speeches
def lemmatise(doc):
    print(f"Lemmatising document...")
    # Remove punctuations, pronounces, and lemmatise all speeches
    spacyDoc = nlp(doc)
    wordList = []
    for token in spacyDoc:
        if token.pos_ != 'PUNCT' and token.lemma_ != '-PRON-':
            wordList.append(token.lemma_.strip())
    doc_lemmatised = " ".join(wordList)
    return doc_lemmatised

# ----- Use multi-processing to performa lemmatisation
startTime = tm.perf_counter()

data = pd.read_csv(f"data/ssm_rel.csv")
speeches = data['Speech']
speechList = speeches.tolist()

speeches_lemma = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    # ----- Using executor.map() for parrallel processing
    # executor.map() applies the function lemmatise() on all elements of and returns a list of results in the same sequence as the inputs
    resultList = executor.map(lemmatise, speechList)

    # Loop thru the list of results
    for result in resultList:
        speeches_lemma.append(result)

# Frame lemmatised speeches and concat with original data
speeches_lemma = pd.DataFrame(speeches_lemma)
data['Speech_lemma'] = speeches_lemma
# Save data
data.to_csv("data/ssm_rel_lemma.csv")

dur = tm.gmtime(tm.perf_counter() - startTime)
print(f"\nLemmatisation complete! Process took {dur.tm_min}min {dur.tm_sec}s")


#%%
