"""
@author: kaisoon
"""
def constructG(DATA, SENTIDIFF_THRES):
    import numpy as np
    import pandas as pd
    import networkx as nx
    import pickle
    import time
    # =====================================================================================
    # ----- FOR DEBUGGING
    DATE = '2017-12'
    PATH = f"results/{DATE}/"

    # PARAMETERS
    # DATA = pd.read_csv(f"{PATH}ssm_{DATE}_results_NMF_senti.csv")
    # with open(f"{PATH}distributions/ssm_{DATE}_sentiDiff.txt", "r") as file:
    #     SENTIDIFF_THRES = int(file.read())


    # =====================================================================================
    # Construct Weighted Graph
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
    for i in range(len(DATA)):
        row_i = DATA.iloc[i]
        # Extract name, topic and sentiment of person1
        p1 = row_i['Person']
        t1 = row_i['Topic_nmf']
        s1 = row_i['Senti_comp']

        for j in range(i+1, len(DATA)):
            row_j = DATA.iloc[j]
            # Extract name, topic and sentiment of person2
            p2 = row_j['Person']
            t2 = row_j['Topic_nmf']
            s2 = row_j['Senti_comp']

            # Print progress...
            if (i % 20 == 0) and (j % 50 == 0):
                print(
                    f"{i:{5}},{j:{5}} of {len(DATA):{5}}\t{p1:{20}}{p2:{20}}\tt1: {int(t1)}\tt2: {int(t2)}")

            # Both actors cannot be the same person
            # Both actors must spoke of the same topic
            # Both sentiment of the same topic must be of the same polarity
            if (p1 != p2) and (t1 == t2) and (s1*s2 > 0):
                # Compute sentiment difference
                sentiDiff = abs(s1 - s2)
                # Both sentiment towards the topic must be less than the threshold
                if (sentiDiff < SENTIDIFF_THRES):
                    # If there is no edge between both actors, construct an edge. Otherwise, update attribtes of the existing edge.
                    if not G.has_edge(p1, p2):
                        agreedSpeeches = {
                            'topic': t1,
                            'sentiDiff': sentiDiff,
                            'data': pd.DataFrame([row_i, row_j])
                        }
                        G.add_edge(p1, p2, weight=1, agreedSpeeches=[agreedSpeeches])
                    else:
                        # Extract data from already existing edge
                        edgeData = G.get_edge_data(p1, p2)

                        # Update weight data
                        weight_old = edgeData['weight']
                        weight_new = weight_old + 1
                        # Update agreedSpeeches
                        agreedSpeeches = edgeData['agreedSpeeches']
                        agreedSpeeches.append(pd.DataFrame([row_i, row_j]))

                        # Update information of the edge
                        G.add_edge(p1, p2, weight=weight_new, agreedSpeeches=agreedSpeeches)
    print('All edges of graph succesfully added!')


    # =====================================================================================
    # Compute centrality and add as node attribute
    cent = {}
    # G.adjacency() returns iterator over nodes and a dict containing names of all nodes adjacent to the node. Value of the dict contains edge attribute between these two nodes.
    for node,adjDict in G.adjacency():
        # Sum up all weight attributes of each edge connected to the node
        edgeWeightSum = 0
        for p in adjDict.keys():
            edgeWeightSum += adjDict[p]['weight']
        cent[node] = edgeWeightSum

    # Place data in dataFrame and sort according to edgeWeightSum
    cent = pd.DataFrame.from_dict(cent, orient='index', columns='Degree'.split())
    cent.sort_values('Degree', ascending=False, inplace=True)
    # Compute degree of centrality with respect to max edgeWeightSum
    cent_max = cent['Degree'].max()
    cent['Normalised2Max'] = cent['Degree']/cent_max

    # Add centrality information to node attribute
    nx.set_node_attributes(G, cent['Normalised2Max'], 'centrality')


    # =====================================================================================
    # Compute cliques and add clique group number as node attribute
    # Construct a dictionary containing cliques within the network labeled by a clique#
    cliques = {}
    num = 0
    for c in nx.find_cliques(G):
        cliques[num] = c
        num += 1

    # For every actor in the network, search all networkCliques to find if the actor is in it
    # Return a dict of actors and the corresponding clique# that the actor is in
    cliqueNum = {}
    actors = np.sort(list(G.node))
    for p in actors:
        inClique = []
        for i in range(len(cliques)):
            if p in cliques[i]:
                inClique.append(i)
        cliqueNum[p] = inClique

    # Add clique information to node attribute
    nx.set_node_attributes(G, cliqueNum, 'cliques')

    print(f"\nGraph construction complete!\nConstruction took {round(time.time()-startTime, 2)}s")
    # Print percentage of edges removed by threshold


    # =====================================================================================
    # Save graph
    nx.write_gpickle(G, f"{PATH}ssm_{DATE}_weightedGraph.gpickle")
    # Save centrality results
    cent.to_csv(f"{PATH}ssm_{DATE}_centrality.csv")
    # Save clique results
    with open(f"{PATH}ssm_{DATE}_cliques.pickle", "wb") as file:
        pickle.dump(cliques, file)

    return G, cent, cliques
