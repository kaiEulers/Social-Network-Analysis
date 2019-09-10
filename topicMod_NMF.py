"""
@author: kaisoon
"""
# TODO: Eventually, save statements have to be removed such that variables are saved outside of function!
# TODO: Model more topics and come up with qualitive description of what each topic is about by looking at the speech with the highest coefficient in that topic
def topicMod(DATA, nTOPICS, nWORDS, COEFFMAX_PERC_THRES, COEFFDIFF_PERC_THRES):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    # ================================================================================
    # ----- FOR DEBUGGING
    DATE = '2017-12'
    PATH = f"results/{DATE}/"

    # PARAMETERS
    # TODO: How do I determine how many topics to model?
    # TODO: How do I determine COEFFMAX_PERC_THRES? WE WANT ABSOLUTE COEFFICIENT THAT ARE HIGH!
    # TODO: How do I determine COEFFDIFF_PERC_THRES? WE WANT COEFF DIFFERENCE THAT ARE HIGH!
    # DATA = pd.read_csv(f"data/ssm_{DATE}_cleaned.csv")
    # nTOPICS = 3
    # nWORDS = 15
    # COEFFMAX_PERC_THRES = 0.1
    # COEFFDIFF_PERC_THRES = 0.1


    # ================================================================================
    # Topic Modelling using Non-negative Matrix Factorisation(NMF)
    # Instantiate Tfidf model
    tfidf = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
    # Instantiate NMF model
    nmf_model = NMF(n_components=nTOPICS)

    # Create document term matrix with tfidf model
    dtm = tfidf.fit_transform(DATA['Speech'])

    # Extract topics from speeches using NMF
    # Apply non-negative matrix factorisation on the document term matrix
    nmf_model.fit(dtm)
    # nmf_model.transform() returns a matrix with coefficients that shows how much each document belongs to each topic
    topicResults_nmf = nmf_model.transform(dtm)

    # Store top w words in dataFrame topics_nmf
    # Number of words that describes topic
    topics_nmf = pd.DataFrame()
    for index,topic in enumerate(nmf_model.components_):
        # Negating an array causes the highest value to be the lowest value and vice versa
        topWordsIndex = (-topic).argsort()[:nWORDS]
        topics_nmf = topics_nmf.append(pd.Series([tfidf.get_feature_names()[i] for i in topWordsIndex]), ignore_index=True)
    topics_nmf = topics_nmf.transpose()


    # ================================================================================
    # To remove ambiguous speeches, the top coefficients are obtained with a certain percentile of this removed. Then the difference between the top-two coefficients are obtained and again, the certain percentile is removed.
    # Sort each row from topResults
    coeff_sorted = [sorted(row) for row in topicResults_nmf]
    # Construct dataFrame with highest coeff and the difference between the top-two coeff
    coeff_stats = {
        'coeffMax': [row[-1] for row in coeff_sorted],
        'coeffDiff': [row[-1] - row[-2] for row in coeff_sorted],
    }
    coeff_stats = pd.DataFrame(coeff_stats, columns='coeffMax coeffDiff'.split())

    # Sort the dataframe by the highest coeff in descending order
    coeff_stats.sort_values(by='coeffMax', inplace=True)
    # Extract coeffMax to be saved later
    coeffMax = pd.DataFrame(coeff_stats['coeffMax'], columns='coeffMax'.split())
    # Only take data above the threshold
    coeffMax_thres = round(len(coeff_stats)*COEFFMAX_PERC_THRES)
    coeff_stats = coeff_stats[coeffMax_thres:]

    # Sort the dataframe by the difference between the top-two coeff in descending order and only take data above the threshold
    coeff_stats.sort_values(by='coeffDiff', inplace=True)
    # Extract coeffDiff to be saved later
    coeffDiff = pd.DataFrame(coeff_stats['coeffDiff'], columns='coeffDiff'.split())
    # Only take data above the threshold
    coeffDiff_thres = round(len(coeff_stats)*COEFFDIFF_PERC_THRES)
    coeff_stats = coeff_stats[coeffDiff_thres:]


    # ================================================================================
    # Retrieve corresponding data from results
    results = DATA.iloc[coeff_stats.index]
    # Assign topic/s to each speech
    topicAssigned_nmf = [topicResults_nmf[i].argmax() for i,row in coeff_stats.iterrows()]

    # Concat assigned topics
    speeches = results['Speech']
    results = results.drop('Speech', axis=1)
    results["Topic_nmf"] = topicAssigned_nmf
    results["Speech"] = speeches
    results.sort_values(by='Speech_id', inplace=True)


    # ================================================================================
    # Analyse results from NMF topic modelling
    # Compute topicCount and percentage of each topic

    topicCount = {t: topicAssigned_nmf.count(t) for t in range(nTOPICS)}
    percentage = {t: round(topicCount[t]/sum(topicCount.values()), 2) for t in range(nTOPICS)}

    topics_nmf.loc['topicCount'] = topicCount
    topics_nmf.loc['percentage'] = percentage

    print(topics_nmf)


    # ================================================================================
    # Save coeffMax
    coeffMax.to_csv(f"{PATH}/distributions/ssm_{DATE}_topicCoeffMax.csv", index=False)
    # Save coeffDiff
    coeffDiff.to_csv(f"{PATH}/distributions/ssm_{DATE}_topicCoeffDiff.csv", index=False)
    # Save results
    results.to_csv(f"{PATH}ssm_{DATE}_results_NMF.csv", index=False)
    # Save analysis of topics
    topics_nmf.to_csv(f"{PATH}ssm_{DATE}_topicsNMF.csv")

    return results, topics_nmf, coeffMax, coeffDiff
