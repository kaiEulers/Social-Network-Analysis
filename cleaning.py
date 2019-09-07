"""
@author: kaisoon
"""
# ----- Imports and Functions
import numpy as np
import pandas as pd
import re

# Function to clean up MP names
def clean_name(name):
    import re
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
    name_cleaned = ' '.join([firstName, lastName])

    return name_cleaned


#%% ---------- Clean Data
FILE_NAME = "marriage_bills_para_short_cleaned.csv"
data = pd.read_csv(f"data/raw/{FILE_NAME}")

data_cleaned = pd.DataFrame(columns='Speech_id Date Bill Type Person Gender Party Elec Metro Speech'.split())
speechIds = data['speech_id'].unique()

# Concatenate all paragraphs of the same speech into one
for sID in speechIds:
    # Extract rows that are of the same speech
    sameSpeech = data[data['speech_id'] == sID]
    # Concat all paragraphs of the speech into one
    fullSpeech = ""
    for p in sameSpeech['para']:
        fullSpeech = fullSpeech + p

    # Replace all "per cent" in speeches with "percent"
    fullSpeech = fullSpeech.replace('per cent', 'percent')

    # Assemble all data for a row
    row = [
        sameSpeech.iloc[0]['speech_id'],
        sameSpeech.iloc[0]['Date'].strip(),
        sameSpeech.iloc[0]['Bill'].strip(),
        sameSpeech.iloc[0]['Type'].strip(),
        clean_name(sameSpeech.iloc[0]['Person'].strip()),
        sameSpeech.iloc[0]['Gender'],
        sameSpeech.iloc[0]['Party'].strip(),
        sameSpeech.iloc[0]['Elec'].strip(),
        sameSpeech.iloc[0]['metro'],
        fullSpeech
    ]
    # Append row to data_cleaned
    data_cleaned = data_cleaned.append(pd.Series(row, index=data_cleaned.columns), ignore_index=True)
    data_cleaned = data_cleaned.sort_values(by=['Date', 'Speech_id'])

data_cleaned.to_csv("data/ssm_cleaned.csv", index=False)


#%% ----- Divide data by the YEAR that the speech was made
FILE_NAME = "ssm_cleaned.csv"
data = pd.read_csv(f"data/{FILE_NAME}")

# Extract all unique years that speeches were made
year = pd.Series([d[:4] for d in data['Date']]).unique()
# Divide speech made in different years in different files
for y in year:
    temp = pd.DataFrame(columns=data.columns)
    for i in data.index:
        if data.iloc[i]['Date'][:4] == y:
            temp = temp.append(data.iloc[i], ignore_index=True)
    temp.to_csv(f"data/{FILE_NAME.replace('cleaned', y + '_cleaned')}", index=False)


#%% ----- Divide data by the MONTH that the speech was made
YEAR = '2017'
FILE_NAME = f"ssm_{YEAR}_cleaned.csv"
data = pd.read_csv(f"data/{FILE_NAME}")

# Extract all unique months that speeches were made
month = pd.Series([d[5:7] for d in data['Date']]).unique()
# Divide speech made in different years in different files
for m in month:
    temp = pd.DataFrame(columns=data.columns)
    for i in data.index:
        if data.iloc[i]['Date'][5:7] == m:
            temp = temp.append(data.iloc[i], ignore_index=True)
    temp.to_csv(f"data/{FILE_NAME.replace('_cleaned', '-' + m + '_cleaned')}", index=False)


#%% ----- Divide data by the DAY that the speech was made
YEAR_MONTH = '2017-12'
FILE_NAME = f"ssm_{YEAR_MONTH}_cleaned.csv"
data = pd.read_csv(f"data/{FILE_NAME}")

# Extract all unique days that speeches were made
day = pd.Series([d[8:10] for d in data['Date']]).unique()
# Divide speech made in different years in different files
for d in day:
    temp = pd.DataFrame(columns=data.columns)
    for i in data.index:
        if data.iloc[i]['Date'][8:10] == d:
            temp = temp.append(data.iloc[i])
    temp.to_csv(f"data/{FILE_NAME.replace('_cleaned', '-' + d + '_cleaned')}", index=False)


#%% ----- Divide results by the DAY the speech was made
YEAR_MONTH = '2017-12'
FILE_NAME = f"ssm_{YEAR_MONTH}_results_NMF_senti.csv"
data = pd.read_csv(f"results/{FILE_NAME}")

# Extract all unique days that speeches were made
day = pd.Series([d[8:10] for d in data['Date']]).unique()
# Divide speech made in different years in different files
for d in day:
    temp = pd.DataFrame(columns=data.columns)
    for i in data.index:
        if data.iloc[i]['Date'][8:10] == d:
            temp = temp.append(data.iloc[i])
    temp.to_csv(f"data/{FILE_NAME.replace('_results', '-' + d + '_results')}", index=False)


#%%
