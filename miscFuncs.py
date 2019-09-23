import datetime
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

def convert2datetime(data):
    """
    :param data: dataframe with 'Date' column as strings
    :return: dataframe with 'Date' column converted to datetime objects
    """
    DATETIME_FORMAT = '%Y-%m-%d'

    # Lambda function that converts a string into a datetime format
    to_datetime = lambda x: datetime.datetime.strptime(x, DATETIME_FORMAT)
    # Apply lambda function to every date string
    dateTime = data['Date'].apply(to_datetime)
    dateTime = dateTime.astype('object')
    # Update data with dates in datetime format
    data.update(dateTime)

    return data

def rearrangeName(name):
    """
    :param name: string with "lastName, firstName, MP"
    :return: string with "firstName lastName"
    """
    # Remove ', MP'
    name = name.replace(', MP', '')

    # Search for first name - all letters after ', '
    firstName = re.search(', .*', name)
    firstName = firstName.group(0)
    # Remove ', ' from firstName
    firstName = firstName.replace(', ', '')

    # Search for last name
    lastName = re.search('.*, ', name)
    lastName = lastName.group(0)
    # Remove ', ' from lastName
    lastName = lastName.replace(', ', '')

    # Join firstName and lastName
    name_rearranged = ' '.join([firstName, lastName])

    return name_rearranged

def crossValid_split(data, k=10):
    """
    :param data:
    :param k: number of fold cross validation
    :return: a generator that contains the data split for k-fold cross validation
    """
    lastIndex = len(data) - 1
    partLength = round(len(data)/k)

    # Construct start and end index for each part to drop from original data
    indexList = [0]
    while lastIndex > 0:
        indexList.append(lastIndex)
        lastIndex -= partLength
    indexList.sort()

    # Yield a generator containing k subsets of data each with a different part dropped from the original data
    for i in range(len(indexList) - 1):
        toDrop = list(range(indexList[i], indexList[i+1]))
        yield data.drop(toDrop)

def similarSum_split(orderedList, N):
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


#%%
