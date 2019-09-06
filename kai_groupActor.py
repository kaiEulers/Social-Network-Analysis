def byAttr(G, attr):
    """
    Returns 2 dictionaries and a list
    :returns a dictionary of each attributes mapped to all actors associated with that attribute
    :returns another dictionary of the colour-map of each attribute. The colour-map is used to colour the network graph.
    :returns a list of attribute names used for the legend of the network graph
    """
    import importlib
    import networkx as nx
    import kai_colourPals as cp
    importlib.reload(cp)

    # ----- Catch invalid input error
    if attr not in 'Party Gender Metro'.split():
        raise ValueError('Attribute must be one of the following: Party, Gender, Metro')


    # ----- Group actors by attribute
    groupedActors = {}
    # Load attributes from graph
    data = nx.get_node_attributes(G, attr)
    # Construct a set of unique groups
    groups = list(set(data.values()))
    groups.sort()

    # For each unique group, place actors with that attribute in the group
    for g in groups:
        # Dictionary comprehension!
        actorsInOneGroup = {k: v for (k, v) in data.items() if v == g}
        groupedActors[g] = list(actorsInOneGroup.keys())


    # ----- Create colourMap base on the grouping for node colouring and legend labels
    if attr == 'Party':
        colourMap = {
            'AG' : cp.sns['green'],
            'ALP' : cp.sns['red'],
            'IND' : cp.sns['brown'],
            'KAP' : cp.sns['purple'],
            'LP' : cp.sns['blue'],
            'Nats' : cp.sns['yellow']
        }
        legend = "Liberal Party of Australia|Australian Labor Party|National Party of Australia|Australian Greens|Independent|Katter's Australian Party".split(
            '|')
        legend.sort()
        return groupedActors, colourMap, legend
    elif attr == 'Gender':
        # Pink for women, blue for men
        colourMap = {
            0 : cp.cbSet1['pink'],
            1 : cp.cbSet1['blue']
        }
        legend = 'Female Male'.split()
        legend.sort()
        return groupedActors, colourMap, legend
    elif attr == 'Metro':
        # The greener the colour, the further away the actor is from the CBD
        colourMap = {
            1 : cp.cbYlGn[0],
            2 : cp.cbYlGn[1],
            3 : cp.cbYlGn[2],
            4 : cp.cbYlGn[3]
        }
        legend = 'Zone 1|Zone 2|Zone 3|Zone 4'.split('|')
        legend.sort()
        return groupedActors, colourMap, legend
    else:
        raise ValueError('Attribute must be one of the following: Party, Gender, Metro')

# TODO: Write function to group actors by centrality. Need to class centrality values into bins.


#%% Check out colour palettes
# import seaborn as sns
# import matplotlib.pyplot as plt
# from palettable.colorbrewer.qualitative import Set1_6
#
# sns.palplot(sns.color_palette().as_hex())
# plt.show()
#
# sns.palplot(sns.color_palette("Paired"))
# plt.show()
#
# sns.palplot(sns.color_palette("Oranges", 4))
# plt.show()
#
# sns.palplot(['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33'])
# plt.show()


#%% Debug colour palettes



#%% Debug grouping actors by attribute
# import networkx as nx
#
# attr = 'Party'
# groupedActors = {}
# # Load attributes from graph
# data = nx.get_node_attributes(G, attr)
# # Construct a set of unique groups
# groups = list(set(data.values()))
# groups.sort()
#
# # For each unique group, place actors with that attribute in the group
# for g in groups:
#     # Dictionary comprehension!
#     actorsInOneGroup = {k: v for (k, v) in data.items() if v == g}
#     groupedActors[g] = list(actorsInOneGroup.keys())