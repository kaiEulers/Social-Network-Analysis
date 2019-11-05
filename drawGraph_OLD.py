"""
@author: kaisoon
"""
import math
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import importlib
import group
import colourPals as cp
import time

def draw(
        G,
        groupBy='party',
        CENT_PERC_THRES=1,
        layout='kamada',
        title='', title_fontsize=None,
        legend=True, legend_font_size=9,
        node_size=10,
        node_alpha=0.85,
        node_linewidth=0.5,
        edge_width=0.25,
        node_label=True, font_size=7,
):
    """
    :param G: is a graph made from networkx
    :param groupBy: 'party'|'gender'|'metro'
    :return:
    """
    importlib.reload(group)
    importlib.reload(cp)
    # ================================================================================
    # ----- FOR DEBUGGING
    # DATE = '2017-12'
    # PATH = f"results/{DATE}/"

    # PARAMETERS
    # G = nx.read_gpickle(f"{PATH}ssm_{DATE}_weightedGraph.gpickle")
    # groupBy = 'party'
    # CENT_PERC_THRES = 0.9
    # layout = 'kamada'; FIG_SIZE = 4
    # title = ''; title_fontsize = None
    # legend = True; legend_font_size = 9
    # node_size = 10; node_size_highCent = 20
    # node_alpha = 0.85
    # node_linewidth = 0.5; node_linewidth_highCent = 2
    # edge_width = 0.25
    # node_label = True; font_size = 7
    # ================================================================================

    # ----- Set up graph layout
    sns.set_style("dark")
    sns.set_context("talk")
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

    # ================================================================================
    # ----- FOR DEBUGGING
    # fig = plt.figure(figsize=(FIG_SIZE*3, FIG_SIZE*2), dpi=300, tight_layout=True)
    # ================================================================================
    # ----- Draw nodes
    plt.title(f"{title}", fontsize=title_fontsize)
    # print("Drawing nodes...")

    # Group nodes by attribute and draw ALL nodes
    parties = nx.get_node_attributes(G, groupBy)
    grouped_party, cMap_nodes, legMap_nodes = group.byNodeAttr(parties, groupBy)
    for grp in grouped_party.keys() :
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=grouped_party[grp],
            node_size=node_size,
            node_color=cMap_nodes[grp],
            alpha=node_alpha,
            edgecolors='black',
            linewidths=node_linewidth,
            label=legMap_nodes[grp]
        )

    # Filter nodes with centrality
    cents = nx.get_node_attributes(G, 'BtwnCent')
    grouped_cent = group.byNodeCent(cents, CENT_PERC_THRES)
    # Reconstruct dictionary for nodes with high centrality
    highCent = {k: v for k, v in parties.items() if k in grouped_cent[1]}
    # Group nodes with high centrality by attribute and draw over already drawn graph
    groupedhighCent_party, cMap_nodesHighCent, _ = group.byNodeAttr(highCent, groupBy)
    for grp in groupedhighCent_party.keys() :
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=groupedhighCent_party[grp],
            node_size=node_size*2,
            node_color=cMap_nodesHighCent[grp],
            edgecolors='black',
            linewidths=node_linewidth*4,
        )
    # print("Node drawing complete!")

    # ================================================================================
    # ----- Draw edges
    # print("Drawing edges...")
    # Retrieve all edges with weights attributes from graph
    weights = nx.get_edge_attributes(G, 'weight')
    # Compute weight relative to max weight
    weight_percentile = {k: v / max(weights.values()) for (k, v) in weights.items()}
    # Group edges by weight
    groupedEdges, cMap_edges, sMap_egdes, legMap_edges = group.byEdgeWeight(weight_percentile)
    for grp in groupedEdges.keys():
        nx.draw_networkx_edges(
            G, pos,
            edgelist=groupedEdges[grp],
            width=edge_width * sMap_egdes[grp],
            edge_color=cMap_edges[grp],
            # label=legMap_edges[grp]
        )
    # print("Edge drawing complete!")

    # ================================================================================
    # ----- Draw node labels
    if node_label:
        # print("Drawing node labels...")
        # Group nodeLabels by actors with high centrality
        groupedLabels, cMap_labels, sMap_labels, fwMap_labels = group.byCent4NodeLabel(cents, CENT_PERC_THRES)
        for grp in groupedLabels.keys():
            nx.draw_networkx_labels(
                G, pos,
                labels=groupedLabels[grp],
                font_size=font_size * sMap_labels[grp],
                font_color=cMap_labels[grp],
                font_weight=fwMap_labels[grp],
            )
        # print("Node label drawing complete!")

    # ----- Draw legend
    if legend:
        plt.legend(markerscale=legend_font_size * 0.05, fontsize=legend_font_size)

    # ================================================================================
    # ----- FOR DEBUGGING
    # fig.savefig(f"{PATH}figures/ssm_{DATE}_graph_by{groupBy.capitalize()}.png", dpi=300)
    # plt.show()
    # ================================================================================

    return
