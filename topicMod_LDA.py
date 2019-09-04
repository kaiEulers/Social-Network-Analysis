#%% -----Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


#%%---------- Topic Modelling using Latent Dirichlet Allocation(LDA)
# ----- Modelling topics for speeches made in the month of 2017-12
nTOPICS = 3
FILE_NAME = "ssm_cleaned_2017-12.csv"
data = pd.read_csv(f"/Users/kaisoon/Google Drive/Code/Python/COMP90019_project/data/{FILE_NAME}")

# Parameter max_df ignore terms that occur over a percentage of the document collection or actual number of time.
# - a floating point number between 0 and 1 will indicate percentage
# - an integer will indicate the actual number of time
# In this case, max_df=0.95 means a term that shows up above 95% of the document collection will be ignored.
# Parameter min_df ignore terms that occur below a percentage of the document collection or actual number of time.
# In this case, min_df=2 means a term that shows up below twice our of all documents will be ignored
# Parameter stop_words ignores stop words base on the language specified.
cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
LDA = LatentDirichletAllocation(n_components=nTOPICS)


# ----- Create document term matrix with countVectorisor
dtm = cv.fit_transform(data['Speech'])


# ----- Extract topics frmo speeches using LDA
# Apply Latent Dirichlet Allocation on the document term matrix
LDA.fit(dtm)
# nmf_model.transform() returns a matrix that indicates the intensity of each document belonging to each topic
topic_results_lda = LDA.transform(dtm)
topic_assigned_lda = pd.DataFrame(topic_results_lda.argmax(axis=1), columns='Topic'.split())

# Store top w words in dataframe topics_lda
# Number of words that describes topic
w = 15
topics_lda = pd.DataFrame()
for index,topic in enumerate(LDA.components_):
    # Negating an array causes the highest value to be the lowest value and vice versa
    topWordsIndex = (-topic).argsort()[:w]
    topics_lda = topics_lda.append(pd.Series([cv.get_feature_names()[i] for i in topWordsIndex]), ignore_index=True)
topics_lda = topics_lda.transpose()


# ----- Concat results
speeches = data['Speech']
topic_assigned_lda = pd.Series(topic_results_lda.argmax(axis=1))
results = data.drop('Speech', axis=1)
results["Topic_lda"] = topic_assigned_lda
results["Speech"] = speeches

# ----- Analyse results from LDA topic modelling
# Check if percentage of SpeechCount of each Topic to determine if the number of topics selected was optimal
# Index of variable analysis is the topic number
analysis_lda = pd.DataFrame(topic_assigned_lda.value_counts().sort_index(), columns='SpeechCount'.split())
analysis_lda["Percentage"] = [round(cnt/sum(analysis_lda["SpeechCount"]), 2) for cnt in analysis_lda["SpeechCount"]]

# Concat analysis_lda to topics_lda
topics_lda = topics_lda.append(analysis_lda["Percentage"])
topics_lda = topics_lda.append(analysis_lda["SpeechCount"])
print(topics_lda)

# Save topics and analysis_lda
topics_lda.to_csv(f"/Users/kaisoon/Google Drive/Code/Python/COMP90019_project/data/results/{re.sub('cleaned', 'topics_LDA', FILE_NAME)}")


#%% Plots
