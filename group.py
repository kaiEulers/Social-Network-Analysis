"""
@author: kaisoon
"""
import colourPals as cp
import importlib
importlib.reload(cp)
# =====================================================================================
def byThres(data, thres):
    grouped = {0: [k for (k, v) in data.items() if v < thres],
               1: [k for (k, v) in data.items() if v >= thres]}

    return grouped


def byOrd4(data):
    grouped = {
        0: [k for k, v in data.items() if 0 < v < 0.25],
        1: [k for k, v in data.items() if 0.5 <= v < 0.75],
        2: [k for k, v in data.items() if 0.75 <= v < 0.875],
        3: [k for k, v in data.items() if v >= 0.875]
    }

    return grouped


# =====================================================================================
def byNodeAttr(data, groupby):
    """
    byNodeAttr() group nodes in a graph according to its attribute
    :param data: is a dict
    :param groupby is a String chosen from the following [Party, Gender, Metro]
    Returns 2 dicts and a list
    :returns grouped is a dict of each attributes mapped to all actors associated with that attribute
    :returns colouMap is a dict of the colour-map of grouped edges. The colour-map is used to colour the network graph.
    :returns legendMap is a dict of legend names used for the network graph
    """
    # ----- Catch invalid input error
    assert groupby in 'party gender metro'.split(), 'Attribute must be one of the following: Party, Gender, Metro'

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
        colourMap = {
            'AG'  : '#39B54A',
            'ALP' : '#DE3533',
            'CA'  : '#FF7400',
            'IND' : cp.cbSet1['brown'],
            'KAP': cp.cbSet1['purple'],
            # 'LP' : '#1B46A5',
            'LP'  : cp.sns['blue'],
            'Nats': '#FEF032',
        }
        legendMap = {
            'AG'  : "Australian Greens",
            'ALP' : "Australian Labor Party",
            'CA'  : "Central Alliance",
            'IND' : "Independent",
            'KAP' : "Katter's Australian Party",
            'LP'  : "Liberal Party of Australia",
            'Nats': "National Party of Australia",
        }
        return grouped, colourMap, legendMap
    elif groupby == 'gender':
        # Using COLOURBREWER SET1 palette for gender. Pink for women, blue for men.
        colourMap = {
            'Female': cp.cbSet2['pink'],
            'Male': cp.cbSet2['blue']
        }
        legendMap = {
            'Female': 'Female',
            'Male': 'Male',
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
            1: 'Zone 1',
            2: 'Zone 2',
            3: 'Zone 3',
            4: 'Zone 4',
        }
        return grouped, colourMap, legendMap
    else:
        raise ValueError('Attribute must be one of the following: Party, Gender, Metro')


# =====================================================================================
def byNodeCent(data, thres):
    """
    byNodeAttr() group nodes in a graph according to its attribute
    :param data: is a dict
    Returns 2 dicts and a list
    :returns grouped is a dict of each attributes mapped to all actors associated with that attribute
    :returns colouMap is a dict of the colour-map of grouped edges. The colour-map is used to colour the network graph.
    :returns legendMap is a dict of legend names used for the network graph
    """
    # ----- Group actors by attribute
    # Split text at centrality threshold
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
    # Group edges by threshold
    grouped = byOrd4(data)

    # Maps colour of edges grouped by weights
    colourMap = {
        0: cp.cbGrays[3],
        1: cp.cbGrays[4],
        2: cp.cbGrays[5],
        3: cp.cbGrays[6],
    }
    # Maps scaling factor of edge widths grouped by weights
    scaleMap = {
        0: 1,
        1: 1.5,
        2: 2,
        3: 3,
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
    :param data: is a dict
    :param thres: is a threshold between the interval [0, 1] at which the text is grouped by
    :return: grouped is a dict. Key0 contains edges above the weight threshold (inclusive). Key1 contains edges below the weight threshold.
    :returns colouMap
    :returns scaleMap
    :returns fontWeightMap
    """
    grouped = byThres(data, thres)
    # Construct dict to reformat names from 'first last' to 'first\nlast'
    grouped[0] = {n: n.replace(' ', '\n') for n in grouped[0]}
    grouped[1] = {n: n.replace(' ', '\n') for n in grouped[1]}

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
