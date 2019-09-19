"""
@author: kaisoon
"""
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import time

# TODO: Eventually, save statements have to be removed such that variables are saved outside of function!
def constructG(DATA, SENTIDIFF_THRES):
    """
    :param DATA: is a DataFrame with columns 'Person, Topic, 'Sentiment', and 'Speech'.
    :param SENTIDIFF_THRES:
    :return:
    """
    # ================================================================================
    # ----- FOR DEBUGGING
    # TIME_FRAME = '2017'
    # METHOD = 'nmf'
    # PATH = f"results/"

    # PARAMETERS
    # DATA = pd.read_csv(f"{PATH}{TIME_FRAME}/ssm_results_{TIME_FRAME}.csv")
    # with open(f"{PATH}ssm_sentiDiffThres_{METHOD}.txt", "r") as file:
    #     SENTIDIFF_THRES = int(file.read())
    # ================================================================================
    # ----- Construct Weighted Graph
    startTime = time.time()
    G = nx.Graph()
    G.clear()

    # ----- Add nodes
    print('\nAdding nodes for graph...')
    for i in DATA.index:
        row = DATA.loc[i]
        person = row['Person']
        # Only add actor if the actor hasn't already been added
        if not G.has_node(person):
            # Construct dataFrame for data attribute of node
            # Extract all data from the actor
            data = DATA[DATA['Person'] == person]
            data.index = range(len(data))

            # Add node with its corresponding attributes
            G.add_node(
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
                    # If there is no edge between both actors, construct an edge. Otherwise, update attribtes of the existing edge.
                    if not G.has_edge(p_i, p_j):
                        agreedSpeeches = {
                            'topic'    : t_i,
                            'sentiDiff': sentiDiff,
                            'data'     : pd.DataFrame([row_i, row_j])
                        }
                        G.add_edge(p_i, p_j, weight=1, agreedSpeeches=[agreedSpeeches])
                    else:
                        # Extract data from already existing edge
                        edgeData = G.get_edge_data(p_i, p_j)

                        # Compute new weight and update weight data
                        weight_old = edgeData['weight']
                        weight_new = weight_old + 1
                        # Contruct new agreedSpeeches dict and append to existing agreedSpeeches
                        agreedSpeeches_old = edgeData['agreedSpeeches']
                        agreedSpeeches_new = [{
                            'topic'    : t_i,
                            'sentiDiff': sentiDiff,
                            'data'     : pd.DataFrame([row_i, row_j])
                        }]
                        agreedSpeeches_new.append(agreedSpeeches_old)

                        # Update information of the edge
                        G.add_edge(p_i, p_j, weight=weight_new, agreedSpeeches=agreedSpeeches_new)
    print('All edges of graph succesfully added!')

    # ================================================================================
    # ----- Compute degree of centrality and add as node attribute
    # Centrality has to be normalised to the max possible number of agreements a node can have
    # This is computed by (number of speeches made by actor)*[(total number of speeches) - (number of speeches made by actor)]
    # G.degree() returns the number of edges adjacent to a node, taking into account of the edge weight
    cent = {n: G.degree(n, weight='weight') for n in list(G.node)}
    cent = pd.DataFrame.from_dict(cent, orient='index', columns='degree'.split())
    # Compute number of speeches each actor have made
    actorSpeechCnt = {}
    for n in list(G.node):
        actorSpeechCnt[n] = len(DATA[DATA['Person'] == n])
    # Compute normalised degree of centrality
    cent_norm = {}
    for n in list(G.node):
        cent_max = actorSpeechCnt[n]*(len(DATA) - actorSpeechCnt[n])
        cent_norm[n] = cent['degree'].loc[n]/cent_max
    cent_norm = pd.DataFrame.from_dict(cent_norm, orient='index', columns='degree_norm'.split())

    # Place normalised data in dataFrame and sort according it
    cent['degree_norm'] = cent_norm
    cent.sort_values(by='degree_norm', ascending=False, inplace=True)

    # Add centrality information to node attribute
    nx.set_node_attributes(G, cent['degree_norm'], 'centrality')

    # ================================================================================
    # ----- Compute cliques and add clique group number as node attribute
    # Construct a dictionary containing cliques within the network labeled by its clique#
    cliques = {i: clq for i, clq in enumerate(nx.find_cliques(G))}

    # For every actor in the network, search all networkCliques to find if the actor is in it
    # Return a dict of actors and the corresponding clique# that the actor is in
    cliqueNum = {}
    actors = np.sort(list(G.node))
    for p in actors:
        inClique = []
        for i, clq in cliques.items():
            if p in clq:
                inClique.append(i)
        cliqueNum[p] = inClique

    # Add clique information to node attribute
    nx.set_node_attributes(G, cliqueNum, 'cliques')

    print(f"\nGraph construction complete!")
    print("Construction took {round(time.time()-startTime, 2)}s")
    print(f"{len(cliques)} cliques found")
    # Print percentage of edges removed by threshold


    # =====================================================================================
    # ----- FOR DEBUGGING
    # # Save results
    # nx.write_gpickle(G, f"{PATH}{TIME_FRAME}/ssm_weightedGraph_{TIME_FRAME}.gpickle")
    # cent.to_csv(f"{PATH}{TIME_FRAME}/ssm_centrality_{TIME_FRAME}.csv")
    # with open(f"{PATH}{TIME_FRAME}/ssm_cliques_{TIME_FRAME}.pickle", "wb") as file:
    #     pickle.dump(cliques, file)
    # ================================================================================

    return G, cent, cliques