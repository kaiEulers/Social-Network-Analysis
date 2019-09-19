"""
@author: kaisoon
"""
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import time

def constructMG(DATA, SENTIDIFF_THRES):
    """
    :param DATA: is a DataFrame with columns 'Person, Topic, 'Sentiment', and 'Speech'.
    :param SENTIDIFF_THRES:
    :return:
    """
    # ================================================================================
    # ----- FOR DEBUGGING
    TIME_FRAME = '2017'
    METHOD = 'nmf'
    PATH = f"results/{TIME_FRAME}/"

    # PARAMETERS
    DATA = pd.read_csv(f"{PATH}ssm_results_{TIME_FRAME}.csv")
    with open(f"{PATH}distributions/ssm_sentiDiffThres_{METHOD}.txt", "r") as file:
        SENTIDIFF_THRES = int(file.read())
    # ================================================================================
    # ----- Construct Weighted Graph
    startTime = time.time()
    MG = nx.MultiGraph()
    MG.clear()

    # ----- Add nodes
    print('\nAdding nodes for graph...')
    for i in DATA.index:
        row = DATA.loc[i]
        person = row['Person']
        # Only add actor if the actor hasn't already been added
        if not MG.has_node(person):
            # Construct dataFrame for data attribute of node
            # Extract all data from the actor
            data = DATA[DATA['Person'] == person]
            data.index = range(len(data))

            # Add node with its corresponding attributes
            MG.add_node(
                person,
                gender=row['Gender'],
                party=row['Party'],
                metro=row['Metro'],
                data=data
            )
            # Print progress...
            if i % 50 == 0:
                print(f"{i:{5}} of {len(DATA):{5}}\t{dict(row['Speech_id Date Person Party'.split()])}")
    print('All nodes of graph succesfully added!')

    # ----- Add edges
    print('\nAdding edges for graph...')
    for i, row_i in DATA.iterrows():
        # Extract name, topic and sentiment of person1
        p_i = row_i['Person']
        t_i = row_i['Topic']
        s_i = row_i['Senti_comp']

        for j, row_j in DATA[:i+1].iterrows():
            # Extract name, topic and sentiment of person2
            p_j = row_j['Person']
            t_j = row_j['Topic']
            s_j = row_j['Senti_comp']

            # Print progress...
            if (i % 20 == 0) and (j % 50 == 0):
                print(
                    f"{i:{5}},{j:{5}} of {len(DATA):{5}}\t{p_i:{20}}{p_j:{20}}\tt_i: {int(t_i)}\tt_j: {int(t_j)}")

            # Both actors cannot be the same person
            # Both actors must spoke of the same topic
            # Both sentiment of the same topic must be of the same polarity
            if (p_i != p_j) and (t_i == t_j) and (s_i*s_j > 0):
                # Compute sentiment difference
                sentiDiff = abs(s_i - s_j)
                # Both sentiment towards the topic must be less than the threshold
                if sentiDiff < SENTIDIFF_THRES:
                    MG.add_edge(p_i, p_j, topic=t_i, sentiDiff=sentiDiff, data=pd.DataFrame([row_i, row_j]))
    print('All edges of graph succesfully added!\n')

    # ----- Compute total pair of speeches that agreed on the same topics
    # Retrieve all agreed topics from multigraph
    agreedTopics = nx.get_edge_attributes(MG, 'topic')
    agreedTopics = list(agreedTopics.values())
    # Retrieve a list of unique topics in the graph
    topics = list(set(agreedTopics))
    topics.sort()
    # Construct a dict containing topic number as key and topic count as value
    topicCount = {t: agreedTopics.count(t) for t in topics}

    # Print topicCounts
    for k, v in topicCount.items():
        print(f"Topic:{k}\t\t{v}")

    # =====================================================================================
    # ----- FOR DEBUGGING
    # Save Multi-graph
    nx.write_gpickle(MG, f"{PATH}ssm_multiGraph.gpickle")
    # Save topicCount
    topicCount = pd.DataFrame.from_dict(topicCount, orient='index', columns='topicCount')
    topicCount.to_csv(f"{PATH}ssm_agreedTopicCount.csv")
    # ================================================================================

    return MG, topicCount
