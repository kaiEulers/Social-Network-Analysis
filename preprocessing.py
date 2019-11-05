"""
@author: kaisoon
"""
# ----- Imports and Functions
import re
import os
import time as tm
import pandas as pd
import concurrent.futures
import spacy

FILE_NAME = "marriage_bills_para_short_cleaned.csv"
data = pd.read_csv(f"data/raw/{FILE_NAME}")
RES = 'para'
# RES = 'speech'
PATH = f"results/resolution_{RES}/"

# ----- Extract all hyphenated words from the dataset
hyphenated = []
for i, speech in data['para'].items():
    for match in re.finditer("\w+(-\w+)+", speech):
        hyphenated.append(match.group(0))
hyphenated = pd.Series(hyphenated)
# Compute the count of each hyphenated word
hyphenated_cnt = hyphenated.value_counts()
hyphenated_cnt.to_csv(f"realityChecks/hypenatedWord_count.csv", header=False)
# Construct list with hyphenated words that occur only once
hyphenated_wordList = hyphenated_cnt[hyphenated_cnt > 1].index.tolist()


def rearrangeName(name):
    """
    :param name: string with "lastName, firstName, MP"
    :return: string with "firstName lastName"
    """
    name = name.strip()
    # Remove ', MP'
    name = name.replace(', MP', '')

    # Search for first name - all letters after ', '
    firstName = re.search(', .*', name)
    firstName = firstName.group(0)
    # Remove ', ' from firstName
    firstName = firstName.replace(', ', '')

    # Search for last name
    lastName = re.search('.*, ', name)
    lastName = lastName.group(0)
    # Remove ', ' from lastName
    lastName = lastName.replace(', ', '')

    # Join firstName and lastName
    name_rearranged = firstName + ' ' + lastName

    return name_rearranged


def sub_hyphen_underscore(text, wordList):
    for hyp_word in wordList:
        if hyp_word in text:
            # print(f"{hyp_word} found in text!")
            replacement = hyp_word.replace("-", "_").lower()
            text = re.sub(hyp_word, replacement, text)
    return text


def cleanPartyName(party):
    party = party.strip()
    party = re.sub("[nN][aA][tT][sS]", "Nats", party)
    party = re.sub("[iI][nN][dD].*", "IND", party)
    party = re.sub("[aA][uU][sS]", "KAP", party)
    return party


def cleanText(text):
    text = text.strip()
    # Replace all "[pP]er cent" in speeches with "percent"
    text = re.sub("[sS]ame.?[sS]ex", "same-sex", text)
    # Replace all "[pP]er cent" in speeches with "percent"
    text = re.sub("[pP]er cent", "percent", text)
    # Replace all "'ve" in speeches with "have"
    text = re.sub("'ve ", " have ", text)
    # Replace all "'ll" in speeches with "will"
    text = re.sub("'ll ", " will ", text)
    # Substitute all hyphens in the hyphenated wordList with an underscore
    text = sub_hyphen_underscore(text, hyphenated_wordList)
    return text


def clean_byPara(data):
    startTime = tm.perf_counter()
    # ----- Concatenate all paragraphs of the same speech into one
    personList = []
    genderList = []
    partyList = []
    paragraphList = []
    # For each row of the text, make adjustments to each feature
    for i, row in data.iterrows():

        # ----- Rearrange each actor's name
        personList.append(rearrangeName(row['Person']))
        # ----- Change gender to Male or Female
        if row['Gender']:
            gender = 'Male'
        else:
            gender = 'Female'
        genderList.append(gender)
        # ----- Standardise party acronym
        party = cleanPartyName(row['Party'])
        partyList.append(party)
        # ----- Adjustments to come words in each paragraph
        para = cleanText(row['para'])
        paragraphList.append(para)

    data_cleaned = pd.DataFrame()
    data_cleaned['Para_id'] = data['para_id']
    data_cleaned['Speech_id'] = data['speech_id']
    data_cleaned['Date'] = data['Date'].apply(lambda x: x.strip())
    data_cleaned['Bill'] = data['Bill'].apply(lambda x: x.strip())
    data_cleaned['Type'] = data['Type'].apply(lambda x: x.strip())
    data_cleaned['Person'] = personList
    data_cleaned['Gender'] = genderList
    data_cleaned['Party'] = partyList
    data_cleaned['Elec'] = data['Elec'].apply(lambda x: x.strip())
    data_cleaned['Metro'] = data['metro']
    data_cleaned['Speech'] = paragraphList

    # Sort and save text
    data_cleaned.sort_values(by=['Date', 'Bill'], inplace=True)

    dur = tm.gmtime(tm.perf_counter() - startTime)
    print(f"Cleaning completed in {dur.tm_sec}s")

    return data_cleaned


def clean_bySpeech(data):
    startTime = tm.perf_counter()
    # ----- Concatenate all paragraphs of the same speech into one
    data_cleaned = pd.DataFrame(columns='Speech_id Date Bill Type Person Gender Party Elec Metro Speech'.split())
    speechIds = data['speech_id'].unique()
    for sID in speechIds:

        # Extract rows that are of the same speech
        sameSpeech = data[data['speech_id'] == sID]
        # Concat all paragraphs of the speech into one
        fullSpeech = ""
        for p in sameSpeech['para']:
            fullSpeech = fullSpeech + p

        # ----- Rearrange each actor's name
        person = rearrangeName(sameSpeech.iloc[0]['Person'].strip())
        # ----- Change gender to Male or Female
        if sameSpeech.iloc[0]['Gender']:
            gender = 'Male'
        else:
            gender = 'Female'
        # ----- Standardise party acronym
        party = cleanPartyName(sameSpeech.iloc[0]['Party'])
        # ----- Adjustments to come words in each paragraph
        fullSpeech = cleanText(fullSpeech)

        # Assemble all text for a row
        row = [
            sameSpeech.iloc[0]['speech_id'],
            sameSpeech.iloc[0]['Date'].strip(),
            sameSpeech.iloc[0]['Bill'].strip(),
            sameSpeech.iloc[0]['Type'].strip(),
            person,
            gender,
            party,
            sameSpeech.iloc[0]['Elec'].strip(),
            sameSpeech.iloc[0]['metro'],
            fullSpeech
        ]
        row = pd.Series(row, index=data_cleaned.columns)
        # Append row to data_cleaned
        data_cleaned = data_cleaned.append(row, ignore_index=True)

    # Sort and save text
    data_cleaned.sort_values(by=['Date', 'Bill'], inplace=True)

    dur = tm.gmtime(tm.perf_counter() - startTime)
    print(f"Cleaning completed in {dur.tm_sec}s")

    return data_cleaned


# ========== Clean text
if RES == 'para':
    data_cleaned = clean_byPara(data)
elif RES == 'speech':
    data_cleaned = clean_bySpeech(data)

data_cleaned.to_csv(f"data/ssm_{RES}Res_cleaned.csv", index=False)
data_cleaned['Speech'].to_csv(f"realityChecks/all{RES.capitalize()}.csv", index=False, header=False)

# ========== Remove Irrelevant Data
# Remove speeches that are first readings
data_rel = data_cleaned[data_cleaned['Type'] != 'First Reading']

# Remove speeches associated with superannuation
index = [i for i, row in data_rel.iterrows() if 'superannuation' not in row['Bill'].lower()]
data_rel = data_rel.loc[index]

# Sort and save text
data_rel.sort_values(by=['Date', 'Bill'], inplace=True)
data_rel.to_csv(f"data/ssm_{RES}Res_rel.csv", index=False)


# ========== Lemmatise speeches
nlp = spacy.load('en_core_web_sm')

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

data = pd.read_csv(f"data/ssm_{RES}Res_rel.csv")
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

# Frame lemmatised speeches and concat with original text
speeches_lemma = pd.DataFrame(speeches_lemma)
data_lemma = data
data_lemma['Speech_lemma'] = speeches_lemma

dur = tm.gmtime(tm.perf_counter() - startTime)
print(f"\nLemmatisation complete! Process took {dur.tm_min}min {dur.tm_sec}s")

# Save text
data_lemma.to_csv(f"data/ssm_{RES}Res_rel_lemma.csv", index=False)
data_lemma['Speech_lemma'].to_csv(f"realityChecks/all{RES.capitalize()}_lemma.csv", index=False, header=False)

os.system('say "Lemmatisation Complete"')


#%%