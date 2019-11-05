import pandas as pd
import kaiFunctions as kf
from importlib import reload
reload(kf)

def actors(results):
    # ================================================================================
    # ----- FOR DEBUGGING
    # from kaiFunctions import load_pickle
    # RES = 'para'
    # METHOD = 'nmf'
    # nTOPICS = 11
    # tf = 'All'
    # PATH = f"results/resolution_{RES}/"
    # results_dict = load_pickle(f"{PATH}ssm_resultsDict_{nTOPICS}topics_{METHOD}")
    # results = results_dict[tf]
    # ================================================================================
    actorList = results['Person'].unique().tolist()
    actorList.sort()
    topicList = results['Topic'].unique().tolist()
    if -1 in topicList:
        topicList.remove(-1)
    topicList.sort()

    actorProfile = pd.DataFrame(columns='Party PartyAgg Gender Metro Elec WordCount SpeechCount'.split())
    topicMatrix = pd.DataFrame(columns=topicList)
    topicMentions = pd.DataFrame(columns=topicList + ['TotalMentions'])
    topicWordCount = pd.DataFrame(columns=topicList)

    for actor in actorList:
        # -----Construct topicMentions of each actor
        # Retrieve all text from one actor
        oneActor = results[results['Person'] == actor]
        speechList = oneActor['Speech_id'].unique().tolist()
        speechList.sort()

        topicList_oneActor = []
        speechCount = 0
        for id in speechList:
            oneSpeech = oneActor[oneActor['Speech_id'] == id]

            # --- topicList_oneSpeech stores all topics spoken in a speech by an actor
            topicList_oneSpeech = oneSpeech['Topic'].unique().tolist()
            if -1 in topicList_oneSpeech:
                topicList_oneSpeech.remove(-1)

            # --- topicList_oneActor stores all topics spoken by an actor in all his/her speeches
            if len(topicList_oneSpeech) > 0:
                topicList_oneActor += topicList_oneSpeech
                speechCount += 1

        topicMentions_oneActor = pd.Series(topicList_oneActor).value_counts()
        topicMentions_oneActor.loc['TotalMentions'] = topicMentions_oneActor.sum()
        topicMentions.loc[actor] = topicMentions_oneActor

        # ----- Construct topic vector for each actor where placeholder represents topic [0, 1, 2, ...]
        topicVector = [0]*len(topicList)
        for topicNum in set(topicList_oneActor):
            topicVector[topicList.index(topicNum)] = 1
        topicMatrix.loc[actor] = topicVector

        # ----- Construct wordCount matrix for each actor
        wordCountList = []
        for topic in topicList:
            oneTopic = oneActor[oneActor['Topic'] == topic]
            wordCount_sum = oneTopic['WordCount'].sum()
            wordCountList.append(wordCount_sum)
        topicWordCount.loc[actor] = wordCountList

        # ----- Gather all results and construct an actor's profile
        if oneActor.iloc[0]['Party'] in 'LP Nats'.split():
            partyAgg = 'Coalition'
        elif oneActor.iloc[0]['Party'] in 'KAP IND CA'.split():
            partyAgg = 'Minor Party'
        elif oneActor.iloc[0]['Party'] == 'ALP':
            partyAgg = 'Labor'
        elif oneActor.iloc[0]['Party'] == 'AG':
            partyAgg = 'Greens'
        # elif oneActor.iloc[0]['Party'] == 'KAP':
        #     partyAgg = 'Katter'
        else:
            partyAgg = oneActor.iloc[0]['Party']
        row = pd.Series({
            'Party': oneActor.iloc[0]['Party'],
            'PartyAgg': partyAgg,
            'Gender'   : oneActor.iloc[0]['Gender'],
            'Metro'    : oneActor.iloc[0]['Metro'],
            'Elec': oneActor.iloc[0]['Elec'],
            'WordCount': sum(wordCountList),
            'SpeechCount': speechCount,
        })
        actorProfile.loc[actor] = row

        # Convert appropriate data to integer type
        actorProfile['Metro'] = actorProfile['Metro'].astype(int)
        actorProfile['WordCount'] = actorProfile['WordCount'].astype(int)
        actorProfile['SpeechCount'] = actorProfile['SpeechCount'].astype(int)
        topicMatrix = topicMatrix.astype(int)
        topicMentions = topicMentions.fillna(0)
        topicMentions = topicMentions.astype(int)


    assert actorProfile.index.unique().tolist() == actorList, "Length of actorProfile & actorList are not the same. Something is wrong!"
    assert topicMatrix.index.unique().tolist() == actorList, "Length of topicMatrix & actorList are not the same. Something is wrong!"
    assert topicMentions.index.unique().tolist() == actorList, "Length of topicMentions & actorList are not the same. Something is wrong!"

    # Rename columns
    # topicMentions.columns = list(map(lambda x: topicLabels[x], topicMentions.columns))
    # Concat topicMentions and topicWordCount to actorProfile
    actorProfile = pd.concat([actorProfile, topicMentions], axis=1)

    return actorProfile, topicMatrix
