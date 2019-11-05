import numpy as np
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment as lsa


def jaccard(list1, list2):
    """
    :param list1: is a list of values
    :param list2: is a list of values
    :return: Jaccard similarity score of both lists
    """
    set1 = set(list1)
    set2 = set(list2)

    I = set1.intersection(set2)
    U = set1.union(set2)
    return len(I)/len(U)


def jaccardAvg(list1, list2):
    """
    :param list1: is a list of values. Both list1 and list 2 have to be of the same length!
    :param list2: is a list of values. Both list1 and list 2 have to be of the same length!
    :return: Average Jaccard similarity score of both lists
    """
    if len(list1) != len(list2):
        raise ValueError("Both list parameters have to be of the same length!")

    # ================================================================================
    # -----For Debugging
    # print(f"list1 length: {len(list1)}")
    # print(f"list2 length: {len(list2)}")
    # ================================================================================

    # n is the number of words in the list
    n = len(list1)
    # Sum all jaccard similarities from comparing a 1 word list to n word list
    jSim_sum = 0
    for d in range(1, len(list1) + 1):
        jSim_sum += jaccard(list1[:d], list2[:d])
    return jSim_sum/len(list1)


def agree(dict1, dict2):
    """
    :param dict1: is a dict containing a set of lists for every key
    :param dict2: is a dict containing a set of lists for every key
    :return: an agreement score as to how both sets are similar to each other
    """
    M = np.empty((len(dict1), len(dict2)))
    M[:] = np.nan

    for k_i, v_i in dict1.items():
        for k_j, v_j in dict2.items():
            M[k_i, k_j] = jaccardAvg(v_i, v_j)

    # M is a 'profit' matrix. It has to be inverted to a 'cost' matrix to find the min
    M_invert = 1 - M
    # Extract the index of the minimum values (hence max value) using the Hungarian Method
    rowIndex, colIndex = lsa(M_invert)
    jSim_max = M[rowIndex, colIndex]

    return sum(jSim_max)/len(dict1)


def cosine(vec1, vec2):
    numerator = np.dot(vec1, vec2)
    denominator = norm(vec1)*norm(vec2)
    return numerator/denominator


# %%
# list1 = 'album, music, best, award, win'.split(', ')
# list2 = 'sport, best, win, medal, award'.split(', ')
#
# listAll1 = [list1[:d] for d in range(1, len(list1)+1)]
# listAll2 = [list2[:d] for d in range(1, len(list2)+1)]
#
# for L1, L2 in zip(listAll1, listAll2):
#     print(jaccardSim(L1, L2))
#     print(jaccardSim_avg(L1, L2))
#     print()

# %%
# dict1 = {
#     0: 'sport, win, award'.split(', '),
#     1: 'bank, finance, money'.split(', '),
#     2: 'music, album, band'.split(', '),
# }
#
# dict2 = {
#     0: 'finance, bank, economy'.split(', '),
#     1: 'music, band, award'.split(', '),
#     2: 'win, sport, money'.split(', '),
# }
#
# print(agree(dict1, dict2))

# %%
vec1 = [1, 0, 0, 0]
vec2 = [1, 1, 1, 1]
cosine(vec1, vec2)
