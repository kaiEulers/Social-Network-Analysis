"""
@author: kaisoon
"""
# TODO: How should I determine the centrality threshold?
def drawGraph(G, groupBy='party', cent_thres=0.75, layout='kamada', title='', legend=True, node_size=10, node_alpha=0.85, node_linewidth=0.5, node_size_highCent= 20, node_linewidth_highCent= 2, font_size=7, edge_width=0.25, legend_font_size=9):
    """
    :param G:
    :param groupBy: 'party'|'gender'|'metro'
    :param cent_thres:
    :param title:
    :param node_size:
    :param node_alpha:
    :param font_size:
    :param edge_width:
    :param legend_font_size:
    :param FIG_SiZE:
    :return:
    """
    # ----- Imports
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx
    import importlib
    import group1
    import colourPals as cp
    import time
    importlib.reload(group1)
    importlib.reload(cp)
    startTime = time.time()

    # ----- Set up graph layout
    sns.set_style("darkgrid")
    sns.set_context("notebook")
    # Get node position in layout
    if layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada':
        pos = nx.kamada_kawai_layout(G, weight=None)
    elif layout == 'kamada_weighted':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'planar':
        pos = nx.planar_layout(G)
    elif layout == 'random':
        pos = nx.random_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'spring':
        pos = nx.spring_layout(G, k=20/math.sqrt(G.number_of_nodes()))
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)


    plt.title(f"{title}")

    # ----- Draw nodes
    print("\nDrawing nodes...")
    # Retrieve all nodes and their centrality
    cents = nx.get_node_attributes(G, 'centrality')
    # Group nodes by centrality
    grouped_cent = group1.byNodeCent(cents, cent_thres)
    # Retrieve all nodes and the party they belong to
    parties = nx.get_node_attributes(G, groupBy)
    # Construct dictionary for nodes grouped by centrality
    notHighCent = {k : v for k, v in parties.items() if k in grouped_cent[0]}
    highCent = {k : v for k, v in parties.items() if k in grouped_cent[1]}
    # Group nodes by party
    grouped_notHighCent, cMap_notHighCent, legMap = group1.byNodeAttr(notHighCent, groupBy)
    grouped_highCent, cMap_highCent, _ = group1.byNodeAttr(highCent, groupBy)
    # Draw edges
    for grp in grouped_notHighCent.keys() :
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=grouped_notHighCent[grp],
            node_size=node_size * 100,
            node_color=cMap_notHighCent[grp],
            alpha=node_alpha,
            edgecolors='black',
            linewidths=node_linewidth,
        )
    for grp in grouped_highCent.keys() :
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=grouped_highCent[grp],
            node_size=node_size_highCent * 100,
            node_color=cMap_highCent[grp],
            alpha=node_alpha,
            edgecolors='black',
            linewidths=node_linewidth_highCent,
        )
    print("Node drawing complete!")


    # ----- Draw node labels
    print("\nDrawing node labels...")
    # Retrieve all nodes and their centrality
    cents = nx.get_node_attributes(G, 'centrality')
    # Group nodeLabels by actors with high centrality
    groupedLabels, cMap_labels, sMap_labels, fwMap_labels = group1.byCent4NodeLabel(cents, cent_thres)
    for grp in groupedLabels.keys():
        nx.draw_networkx_labels(
            G, pos,
            labels=groupedLabels[grp],
            font_size=font_size * sMap_labels[grp],
            font_color=cMap_labels[grp],
            font_weight=fwMap_labels[grp],
        )
    print("Node label drawing complete!")


    # ----- Draw edges
    print("\nDrawing edges...")
    # Retrieve all edges with weights attributes from graph
    weights = nx.get_edge_attributes(G, 'weight')
    # Compute weight relative to max weight
    weight_relMax = {k : v / max(weights.values()) for (k, v) in weights.items()}
    # Group edges by weight
    groupedEdges, cMap_edges, sMap_egdes, legMap_edges = group1.byEdgeWeight(weight_relMax)
    # Draw edges
    for grp in groupedEdges.keys():
        nx.draw_networkx_edges(
            G, pos,
            edgelist=groupedEdges[grp],
            width=edge_width * sMap_egdes[grp],
            edge_color=cMap_edges[grp],
        )
    print("Edge drawing complete!")


    # ----- Draw legend
    # TODO: Legend is still a problem if ndoes are grouuped by centrality
    if legend == True:
        plt.legend(markerscale=legend_font_size * 0.05, fontsize=legend_font_size)


    print(f"\nDrawing completed in {time.time()-startTime}s!")

    return