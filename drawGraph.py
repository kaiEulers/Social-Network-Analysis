"""
@author: kaisoon
"""
def draw(G, GROUPBY='party', CENT_THRES=0.75, TITLE='', NODE_SIZE=11, NODE_ALPHA = 0.85, NODE_LABEL_SIZE = 7, EDGE_WIDTH = 0.25, LEGEND_FONT_SIZE = 9, FIG_SiZE = 4):
    """
    :param G:
    :param GROUPBY: 'party'|'gender'|'metro'
    :param CENT_THRES:
    :param TITLE:
    :param NODE_SIZE:
    :param NODE_ALPHA:
    :param NODE_LABEL_SIZE:
    :param EDGE_WIDTH:
    :param LEGEND_FONT_SIZE:
    :param FIG_SiZE:
    :return:
    """
    # ----- Imports
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx
    import importlib
    import group
    import colourPals as cp
    import time
    importlib.reload(group)
    importlib.reload(cp)

    # ----- Set up graph layout
    sns.set_style("darkgrid")
    sns.set_context("notebook")
    # Get node position in kamada kawai layout
    # TODO: Try different layouts
    pos = nx.kamada_kawai_layout(G, weight=None)

    # ----- Draw Network
    startTime = time.time()
    print('Drawing graph...')

    # ----- Draw nodes
    print("\nDrawing nodes...")
    fig = plt.figure(figsize=(FIG_SiZE*3, FIG_SiZE*2), dpi=300, tight_layout=True)
    plt.title(f"{TITLE}")

    # Group by actors by party for node colouring
    groupedActors, cm_nodes, leg_nodes = group.byNodeAttr(G, GROUPBY)
    for grp in groupedActors.keys():
        nx.draw_networkx_nodes(G, pos, nodelist=groupedActors[grp], node_size=NODE_SIZE*100, node_color=cm_nodes[grp], alpha=NODE_ALPHA, edgecolors='black', linewidths=0.5, label=leg_nodes[grp])
    print("Node drawing complete!")


    # ----- Draw node labels
    print("\nDrawing node labels...")
    # Group nodeLabels by actors with high centrality
    groupedLabels, cm_labels, sm_labels, fwm_labels = group.byCent4NodeLabel(G, CENT_THRES)
    for grp in groupedLabels.keys():
        nx.draw_networkx_labels(G, pos, labels=groupedLabels[grp], font_size=NODE_LABEL_SIZE*sm_labels[grp], font_color=cm_labels[grp], font_weight=fwm_labels[grp])
    print("Node label drawing complete!")


    # ----- Draw edges
    print("\nDrawing edges...")
    groupedEdges, cm_edges, sm_egdes, leg_edges = group.byEdgeWeight(G)
    for grp in groupedEdges.keys():
        nx.draw_networkx_edges(G, pos, edgelist=groupedEdges[grp], width=EDGE_WIDTH*sm_egdes[grp], edge_color=cm_edges[grp], label=leg_edges[grp])
    print("Edge drawing complete!")


    # ----- Draw legend
    plt.legend(markerscale=LEGEND_FONT_SIZE*0.05, fontsize=LEGEND_FONT_SIZE)
    print(f"\nGraph drawing complete! Drawing took {time.time()-startTime}s!")

    return fig
