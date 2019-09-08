"""
@author: kaisoon
"""
# TODO: How should I determine the centrality threshold?
def drawGraph(G, groupBy='party', cent_thres=0.75, layout='kamada', title='', legend=True, node_size=11, node_alpha=0.85, node_linewidth=0.5, font_size=7, edge_width=0.25, legend_font_size=9):
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
    import group
    import colourPals as cp
    import time
    importlib.reload(group)
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
    # Extract nodes with centrality above the threshold
    cent = nx.get_node_attributes(G, 'centrality')
    highCent = [k for k, v in cent.items() if v > cent_thres]
    # Group by actors by party for node colouring
    groupedNodes, cMap_nodes, legMap_nodes = group.byNodeAttr(G, groupBy)
    for grp in groupedNodes.keys():
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=groupedNodes[grp],
            node_size=node_size * 100,
            node_color=cMap_nodes[grp],
            alpha=node_alpha,
            edgecolors='black',
            linewidths=node_linewidth,
            label=legMap_nodes[grp]
        )
    print("Node drawing complete!")


    # ----- Draw node labels
    print("\nDrawing node labels...")
    # Group nodeLabels by actors with high centrality
    groupedLabels, cMap_labels, sMap_labels, fwMap_labels = group.byCent4NodeLabel(G, cent_thres)
    for grp in groupedLabels.keys():
        nx.draw_networkx_labels(
            G, pos,
            labels=groupedLabels[grp],
            font_size=font_size * sMap_labels[grp],
            font_color=cMap_labels[grp],
            font_weight=fwMap_labels[grp])
    print("Node label drawing complete!")


    # ----- Draw edges
    print("\nDrawing edges...")
    groupedEdges, cMap_edges, sMap_egdes, legMap_edges = group.byEdgeWeight(G)
    for grp in groupedEdges.keys():
        nx.draw_networkx_edges(
            G, pos,
            edgelist=groupedEdges[grp],
            width=edge_width * sMap_egdes[grp],
            edge_color=cMap_edges[grp],
            label=legMap_edges[grp])
    print("Edge drawing complete!")


    # ----- Draw legend
    if legend == True:
        plt.legend(markerscale=legend_font_size * 0.05, fontsize=legend_font_size)


    print(f"\nDrawing completed in {time.time()-startTime}s!")

    return