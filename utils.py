import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import math
import numpy as np
import random
import datetime
import sys
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import scipy

import torch.nn.functional as F

random.seed(42)

# function: parses date string into DateTime Object
# input: date
# output: DateTime Object
def dateParser(date):
    formats = ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M', '%m/%d/%y %H:%M']
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date, fmt).timestamp()
        except ValueError:
            pass
        except TypeError:
            pass
    return np.nan

# function: get all aggregate glucose stats accessible by dictionary
# input: glucose values (in a time span)
# output: dictionary of type of summary and value
def createGlucStats(glucose):
    d = {}
    d['mean'] = np.mean(glucose)
    d['std'] = np.std(glucose)
    d['min'] = np.min(glucose)
    d['max'] = np.max(glucose)
    d['q1'] = np.percentile(glucose, 25)
    d['q3'] = np.percentile(glucose, 75)
    d['skew'] = scipy.stats.skew(glucose)
    return d
