
from cProfile import label
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd
from __init__ import rlr_validate


def basicStatistics(data):
    """Return the mean, median, and standard deviation of the data."""
    mean = np.mean(data, axis=0)
    median = np.median(data, axis=0)
    std = np.std(data, axis=0)
    return mean, median, std


def normalizeData(data):
    """Normalize the data to zero mean."""
    mean, _, std = basicStatistics(data)
    data = (data - mean)/std
    return data

def transformData(data):
    """Convert text to numbers."""
    data['famhist'] = data['famhist'].map({'Present': 1, 'Absent': 0})
    data.drop('chd', axis=1, inplace=True)
    data.drop('row.names', axis=1, inplace=True)
    data.drop('famhist', axis=1, inplace=True)
    y = data['ldl'].squeeze()
    data.drop('ldl', axis=1, inplace=True)
    attributeNames = data.columns
    return np.array(data), np.array(y), np.array(attributeNames)
