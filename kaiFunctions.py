from copy import copy
import datetime
import numpy as np


def log(text, path):
    with open(f"{path}.log", 'a') as file_handler:
        file_handler.write(f"{text}\n")


def save_pickle(object, path):
    import pickle
    with open(f"{path}.pickle", 'wb') as file_handler:
        pickle.dump(object, file_handler)


def load_pickle(path):
    import pickle
    with open(f"{path}.pickle", 'rb') as file_handler:
        return pickle.load(file_handler)


def convert2datetime(data_in):
    """
    :param data_in: dataframe with 'Date' column as strings
    :return: dataframe with 'Date' column converted to datetime objects
    """
    DATETIME_FORMAT = '%Y-%m-%d'

    # Lambda function that converts a string into a datetime format
    to_datetime = lambda x: datetime.datetime.strptime(x, DATETIME_FORMAT)
    # Apply lambda function to every date string
    dateTime = data_in['Date'].apply(to_datetime)
    dateTime = dateTime.astype('object')
    # Update text with timeframes in datetime format
    data_out = copy(data_in)
    data_out.update(dateTime)

    return data_out


def crossValid_split(data, k=10):
    """
    :param data:
    :param k: number of fold cross validation
    :return: a generator that contains the text split for k-fold cross validation
    """
    lastIndex = len(data) - 1
    partLength = round(len(data)/k)

    # Construct start and end index for each part to drop from original text
    indexList = [0]
    while lastIndex > 0:
        indexList.append(lastIndex)
        lastIndex -= partLength
    indexList.sort()

    # Yield a generator containing k subsets of text each with a different part dropped from the original text
    for i in range(len(indexList) - 1):
        toDrop = list(range(indexList[i], indexList[i + 1]))
        yield data.drop(toDrop)


def split_similarSum(orderedList, N):
    """
    If ordered list is [1, 2, 3, 4] and N=2 then [4,1] and [3,2] will be returned
    If ordered list is [1, 2, 3, 4] and N=3 then [4], [3], and [2,1] will be returned
    :param orderedList:
    :param N: Number of chunks to split the list into
    :return: N chunks of the orderedList such that the summation of all values in each chunk is almost equal
    """
    # Sort list first, just in case it is not ordered
    orderedList.sort()

    # Construct N list of chunks
    chunkList = [[] for _ in range(N)]
    chunkSize = int(np.ceil(len(orderedList)/N))
    # k starts from the last element of the list
    k = len(orderedList) - 1

    for i in range(chunkSize):
        # switch alternates between 0 and 1
        switch = i%2
        if not switch:
            for chunk in chunkList:
                if k < 0:
                    break
                chunk.append(orderedList[k])
                k -= 1
        else:
            for chunk in reversed(chunkList):
                if k < 0:
                    break
                chunk.append(orderedList[k])
                k -= 1
    return chunkList


def get_outliers(
        data, STD_NORM, side, METHOD='yeo-johnson',
        PLOT=False, title=None, title_fontsize=None,
        x_label=None, y_label=None, label_fontsize=None
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import PowerTransformer
    from statsmodels.graphics.gofplots import qqplot
    import colourPals as cp
    import importlib
    importlib.reload(cp)
    # ==================================================
    # Error checking
    assert side == 'left' or side == 'right', "'side' argument has to be either 'left' or 'right'"
    # ==================================================
    # If minimum text is less than zero, and 'box-cox' is selected, compute constant k to shift the text cos that the transformation can be performed.
    if METHOD == 'box-cox' and min(data) <= 0:
        k = 1 - min(data)
        data = data + k

    # ----- Transform text
    pt = PowerTransformer(method=METHOD)
    # Find optimal lambda value for transform
    pt.fit(data.to_numpy().reshape(-1, 1))
    # Transform text to a normal distribution
    data_trans = pt.transform(data.to_numpy().reshape(-1, 1))

    # ----- Compute threshold to remove text above or below threshold
    data_trans_thres = data_trans.mean() + STD_NORM*data_trans.std()
    # Transform threshold back to original distribution
    data_thres = pt.inverse_transform(np.array(data_trans_thres).reshape(1, -1))
    data_thres = data_thres.flatten()[0]

    # If text was shifted before, shift the text back by the same constant.
    if 'k' in locals():
        data_thres = data_thres - k
        data = data - k

    # If normalised standard deviation is less than 0, remove negative end of the text.
    # If normalised standard deviation is more than or equal to 0, remove positive end of the text.
    if side == 'left':
        outliers = data[data < data_thres]
    elif side == 'right':
        outliers = data[data > data_thres]
    else:
        raise ValueError("Argument side has to be 'left'or 'right' ")

    # Flatten can covert transformed text to a series
    data_trans = pd.Series(data_trans.flatten())

    if PLOT:
        FIG_SIZE = 3
        sns.set_style("darkgrid")
        sns.set_context("notebook")
        fig, ax = plt.subplots(nrows=3, figsize=(FIG_SIZE*2, FIG_SIZE*3), dpi=300)

        # Plot coeffMax before transformation
        sns.distplot(data, rug=True, kde=False, ax=ax[0], color=cp.cbPaired['blue'])
        ax[0].axvline(x=data_thres, c=cp.cbPaired['red'])
        ax[0].set_title(title, fontsize=title_fontsize)
        ax[0].set_xlabel(x_label, fontsize=label_fontsize)
        ax[0].set_ylabel(f"Frequency", fontsize=label_fontsize)

        # Plot coeffMax after transformation
        sns.distplot(data_trans, rug=True, kde=False, ax=ax[1], color=cp.cbPaired['purple'])
        ax[1].axvline(x=data_trans_thres, c=cp.cbPaired['red'])
        ax[1].set_xlabel(f"{METHOD.capitalize()} Transformed", fontsize=label_fontsize)
        ax[1].set_ylabel(f"Frequency", fontsize=label_fontsize)

        # Plot qqplot of coeffMax after transformation
        qqplot(data_trans, ax=ax[2], line='s', color=cp.cbPaired['purple'])

        plt.tight_layout()
        plt.show()

    return outliers, data_thres


def parse(input):
    import re
    if type(input) == list:
        return ' '.join(map(str, input))
    elif type(input) == str:
        return list(map(int, input.split()))
    else:
        raise ValueError("input must be either a list of integers or a string of integers")


