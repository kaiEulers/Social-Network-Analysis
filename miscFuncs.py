import datetime
import re

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
