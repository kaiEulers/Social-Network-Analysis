"""
@author: kaisoon
"""
from copy import copy
import pandas as pd
import networkx as nx
import similarity as simi
# ================================================================================
# ----- FOR DEBUGGING
# TOPIC_LABELS = {
#     # Pro-SSM
#     0: "Love\n&\nEquality",
#     # Pro & Anti SSM
#     1: "Protecting\nReligious\nFreedom",
#     # Anti-SSM
#     2: "Institution\nof\nMarriage",
#     # Pro & Anti SSM
#     7: "Discrimination\nof\nSame-sex\nCouples",
#     # Pro & Anti SSM
#     10: "Parents,\nChildren,\n&\nFamily",
# }
# TERM = '45th Parliament'
# actorProfile = actorProfile_dict[TERM]
# topicMatrix = topicMatrix_dict[TERM]
# threshold = 0
# metric = 'cosine'
#%% ================================================================================
def topicProj(topicMatrix, threshold=0):
    G = nx.Graph()
    G.clear()
    # ----- Add nodes
    # print('\nAdding nodes for graph...')
    for topic, _ in topicMatrix.iteritems():
        # Extract all text from the actor
        G.add_node(topic)
    # print('All nodes successfully added!')

    # ----- Add edges
    # print('\nAdding edges for graph...')
    for i, topic_i in enumerate(topicMatrix.columns):
        for j, topic_j in enumerate(topicMatrix.columns[(i+1):]):
            similarity = simi.cosine(topicMatrix[topic_i], topicMatrix[topic_j])
            if similarity > threshold:
                G.add_edge(topic_i, topic_j, weight=similarity)
    # print('All edges successfully added!')

    cent = [G.degree(topic, weight='weight') for topic in topicMatrix.columns]
    cent = pd.DataFrame(cent, columns='edgeWeightSum'.split(), index=topicMatrix.columns)
    # Normalise by the max number of other node a node can be connected to
    cent['Degree'] = cent['edgeWeightSum']/(len(topicMatrix.columns) - 1)

    # Add centrality information to node attribute
    nx.set_node_attributes(G, cent['Degree'], 'degreeCent')

    return G, cent


#%%
def actorProj(actorProfile, topicMatrix, threshold=0):
    # ================================================================================
    # ----- Error Checking
    assert (actorProfile.index).equals(topicMatrix.index), "actorProfile and topicMatrix do not have the same indicies."
    # ================================================================================
    # ----- Construct Weighted Graph
    G = nx.Graph()
    G.clear()

    # ----- Add nodes
    # print('\nAdding nodes for graph...')
    for actor, row in actorProfile.iterrows():
        # Extract all text from the actor
        G.add_node(
            actor,
            gender=row['Gender'],
            party=row['Party'],
            metro=row['Metro'],
            elec=row['Elec'],
            wordCount=row['WordCount'],
        )
    # print('All nodes successfully added!')

    # ----- Add edges
    # print('\nAdding edges for graph...')
    for i, (a_i, row_i) in enumerate(actorProfile.iterrows()):
        for j, (a_j, row_j) in enumerate(actorProfile[(i+1):].iterrows()):
            # Retrieve and parse topicVector profile of actor_i and actor_j
            tv_i = topicMatrix.loc[a_i].tolist()
            tv_j = topicMatrix.loc[a_j].tolist()
            similarity = simi.cosine(tv_i, tv_j)
            # Both actors cannot be the same person and similarity is above a threshold
            if a_i != a_j and similarity > threshold:
                # Add edge with appropriate attributes
                G.add_edge(
                    a_i, a_j,
                    weight=similarity,
                    closeness=abs(1-similarity),
                )
                # Log progress...
                # logPath = "jaccardSim"
                # kf.log(f"{i:{5}},{j:{5}} of {len(actorProfile):{5}}", logPath)
                # kf.log(f"{a_i:{30}}{a_j}", logPath)
                # kf.log(f"{row_i['Topics']:{30}}{row_i['Topics']}", logPath)
                # kf.log(f"Jaccard Similarity: {similarity}\n", logPath)
    # print('All edges successfully added!')


    # ================================================================================
    # ----- Compute degree of centrality and add as node attribute
    # G.degree() returns the number of edges adjacent to a node, taking into account of the edge weight
    cent = pd.DataFrame(
        [G.degree(actor, weight='weight') for actor in actorProfile.index],
        columns='edgeWeightSum'.split(), index=actorProfile.index)
    # Normalise by the max number of other node a node can be connected to
    cent['Degree'] = cent['edgeWeightSum']/(len(actorProfile) - 1)
    cent['Betweenness'] = pd.Series(nx.betweenness_centrality(G))
    # cent['Betweenness'] = pd.Series(nx.betweenness_centrality(G, weight='weight'))
    # cent['Betweenness'] = pd.Series(nx.betweenness_centrality(G, weight='closeness'))
    cent['Closeness'] = pd.Series(nx.closeness_centrality(G))
    # cent['Closeness'] = pd.Series(nx.closeness_centrality(G, distance='weight'))
    # cent['Closeness'] = pd.Series(nx.closeness_centrality(G, distance='closeness'))

    # Add centrality information to node attribute
    nx.set_node_attributes(G, cent['Degree'], 'DegreeCent')
    nx.set_node_attributes(G, cent['Betweenness'], 'BtwnCent')
    nx.set_node_attributes(G, cent['Closeness'], 'ClosenessCent')

    # Find list of connected componenets
    connected_components = list(nx.connected_components(G))

    # Concat centrality measures to actorprofile
    actorProfile_out = copy(actorProfile)
    actorProfile_out['DegreeCentrality'] = cent['Degree']
    actorProfile_out['BtwnCentrality'] = cent['Betweenness']
    actorProfile_out['ClosenessCentrality'] = cent['Closeness']

    # =====================================================================================
    # ----- FOR DEBUGGING
    # # Save results
    # nx.write_gpickle(G, f"{PATH}{TIME_FRAME}/ssm_weightedGraph_{TIME_FRAME}.gpickle")
    # cent.to_csv(f"{PATH}{TIME_FRAME}/ssm_centrality_{TIME_FRAME}.csv")
    # with open(f"{PATH}{TIME_FRAME}/ssm_cliques_{TIME_FRAME}.pickle", "wb") as file:
    #     pickle.dump(cliques, file)
    # ================================================================================

    return G, actorProfile_out, connected_components
