"""
@author: kaisoon
"""
# Functions goes with attempt of sorting nodes by centrality and party
# =====================================================================================
def byThres(data, thres):
    """
    byThres() groups data by a specified threshold
    :param data: is a dict
    :param thres: is a threshold between the interval [0, 1] at which the data is grouped by
    :return: grouped is a dict where key0 contains values above the threshold (inclusive) and key1 contains values below the threshold.
    """
    grouped = {}
    # Find keys below the weight threshold
    # grouped[0] = [k for (k, v) in data.items() if v < thres]
    grouped[0] = [k for (k, v) in data.items() if v < thres]
    # Find keys above the weight threshold
    # grouped[1] = [k for (k, v) in data.items() if v >= thres]
    grouped[1] = [k for (k, v) in data.items() if v >= thres]

    return grouped

def byOrd5(data):
    """
    bySeq5() groups data in five sequential classes
    :param data: is a dict
    :return: grouped is a dict
    """
    grouped = {}
    grouped[0] = [k for k, v in data.items() if v >= 0 and v < 0.25]
    grouped[1] = [k for k, v in data.items() if v >= 0.25 and v < 0.5]
    grouped[2] = [k for k, v in data.items() if v >= 0.5 and v < 0.75]
    grouped[3] = [k for k, v in data.items() if v >= 0.75]

    return grouped

# =====================================================================================
def byNodeAttr(data, groupby):
    """
    byNodeAttr() group nodes in a graph according to its attribute
    :param G is a graph constructed with networkx
    :param groupby is a String chosen from the following [Party, Gender, Metro]
    Returns 2 dicts and a list
    :returns grouped is a dict of each attributes mapped to all actors associated with that attribute
    :returns colouMap is a dict of the colour-map of grouped edges. The colour-map is used to colour the network graph.
    :returns legendMap is a dict of legend names used for the network graph
    """
    import colourPals as cp
    import importlib
    importlib.reload(cp)

    # ----- Catch invalid input error
    if groupby not in 'party gender metro'.split():
        raise ValueError('Attribute must be one of the following: Party, Gender, Metro')


    # ----- Group actors by attribute
    grouped = {}
    # Construct a set of unique groups
    groups = list(set(data.values()))
    groups.sort()

    # For each unique group, place actors with that attribute in the group
    for g in groups:
        grouped[g] = [k for (k, v) in data.items() if v == g]


    # ----- Create colourMap base on the grouping for node colouring and legend labels
    if groupby == 'party':
        # Using SEABORN default palette for parties
        # colourMap = {
        #     'AG': cp.sns['green'],
        #     'ALP': cp.sns['red'],
        #     'IND': cp.sns['brown'],
        #     'KAP': cp.sns['purple'],
        #     'LP': cp.sns['blue'],
        #     'Nats': cp.sns['yellow'],
        # }
        colourMap = {
            'AG': cp.cbSet1['green'],
            'ALP': cp.cbSet1['red'],
            'IND': cp.cbSet1['brown'],
            'KAP': cp.cbSet1['purple'],
            'LP': cp.cbSet1['blue'],
            'Nats': cp.cbSet1['yellow'],
        }
        legendMap = {
            'AG' : "Australian Greens",
            'ALP' : "Australian Labor Party",
            'IND' : "Independent",
            'KAP' : "Katter's Australian Party",
            'LP' : "Liberal Party of Australia",
            'Nats' : "National Party of Australia",
        }
        return grouped, colourMap, legendMap
    elif groupby == 'gender':
        # Using COLOURBREWER SET1 palette for gender. Pink for women, blue for men.
        colourMap = {
            0: cp.cbSet1['pink'],
            1: cp.cbSet1['blue']
        }
        legendMap = {
            0: 'Female',
            1: 'Male',
        }
        return grouped, colourMap, legendMap
    elif groupby == 'metro':
        # Using COLOURBREWER YlGn palette for metro. The greener the colour, the further away the actor is from the CBD.
        colourMap = {
            1: cp.cbYlGn[0],
            2: cp.cbYlGn[1],
            3: cp.cbYlGn[2],
            4: cp.cbYlGn[3]
        }
        legendMap = {
            1 : 'Zone 1',
            2 : 'Zone 2',
            3 : 'Zone 3',
            4 : 'Zone 4',
        }
        return grouped, colourMap, legendMap
    else:
        raise ValueError('Attribute must be one of the following: Party, Gender, Metro')


# =====================================================================================
def byNodeCent(data, thres):
    """
    byNodeAttr() group nodes in a graph according to its attribute
    :param G is a graph constructed with networkx
    Returns 2 dicts and a list
    :returns grouped is a dict of each attributes mapped to all actors associated with that attribute
    :returns colouMap is a dict of the colour-map of grouped edges. The colour-map is used to colour the network graph.
    :returns legendMap is a dict of legend names used for the network graph
    """
    import colourPals as cp
    import importlib
    importlib.reload(cp)

    # ----- Group actors by attribute
    # Split data at centrality threshold
    grouped = byThres(data, thres)

    return grouped


def byEdgeWeight(data):
    """
    byEdgeWeight() group edges in a graph according to its weight
    :param G: is a graph constructed with networkx
    :return: grouped is a dict
    :returns colouMap
    :returns scaleMap
    :returns legendMap
    """
    import colourPals as cp
    import importlib
    importlib.reload(cp)

    # Group edges by threshold
    grouped = byOrd5(data)

    # Maps colour of edges grouped by weights
    colourMap = {
        0: cp.cbGrays[5],
        1: cp.cbGrays[6],
        2: cp.cbGrays[7],
        3: cp.cbGrays[8],
    }
    # Maps scaling factor of edge widths grouped by weights
    scaleMap = {
        0: 1,
        1: 2,
        2: 4,
        3: 7,
    }
    legendMap = {
        0: '1st Quartile',
        1: '2nd Quartile',
        2: '3rd Quartile',
        3: '4th Quartile',
    }

    return grouped, colourMap, scaleMap, legendMap


# =====================================================================================
def byCent4NodeLabel(data, thres):
    """
    byNodeCent() group nodes in a graph according to its centrality
    :param G: is a graph constructed with networkx
    :param thres: is a threshold between the interval [0, 1] at which the data is grouped by
    :return: grouped is a dict. Key0 contains edges above the weight threshold (inclusive). Key1 contains edges below the weight threshold.
    :returns colouMap
    :returns scaleMap
    :returns fontWeightMap
    """
    import colourPals as cp
    import importlib
    importlib.reload(cp)

    grouped = byThres(data, thres)
    # Construct dict to reformat names from 'first last' to 'first\nlast'
    low = {n : n.replace(' ', '\n') for n in grouped[0]}
    high = {n : n.replace(' ', '\n') for n in grouped[1]}
    grouped[0] = low
    grouped[1] = high

    # Maps colour of node label
    colourMap = {
        0: cp.cbGrays[8],
        1: cp.cbGrays[8],
    }
    # Maps scaling factor of nodeLabel
    scaleMap = {
        0: 1,
        1: 1.25,
    }
    # Maps font weight of nodeLabel
    fontWeightMap = {
        0: 'normal',
        1: 'bold',
    }
    return grouped, colourMap, scaleMap, fontWeightMap

