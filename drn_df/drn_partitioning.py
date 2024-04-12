
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from typing import Callable, Optional
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


from tqdm.auto import trange, tqdm
from typing import List, Union

import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution, constraints
from torch.distributions.mixture_same_family import MixtureSameFamily

import re

import sklearn
import scipy
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import quad
from scipy.stats import norm

def uniform_cutpoints(c_0, c_K, p, y):
    num_cutpoints = int(np.ceil(p * len(y)))
    cutpoints = list(np.linspace(c_0, c_K, num_cutpoints))

    return(cutpoints)
    
def merge_cutpoints(cutpoints, y, min_obs):
    # Ensure cutpoints are sorted and unique to start with
    cutpoints = np.unique(cutpoints)
    cutpoints.sort()
    
    if len(cutpoints) == 0:
        return []


    # Initialize
    new_cutpoints = [cutpoints[0]]  # Start with the first cutpoint
    count_since_last_added = 0      # Initialize count of observations since last added cutpoint
    
    for i in range(1, len(cutpoints)):
        # Count observations between this and the previous cutpoint
        observations = np.sum((y >= cutpoints[i - 1]) & (y < cutpoints[i]))
        count_since_last_added += observations

        # If the count reaches or exceeds 5, or we are at the last cutpoint, add the cutpoint to new_cutpoints
        if count_since_last_added >= min_obs or i == len(cutpoints) - 1:
            if cutpoints[i] < np.max(y) or i == len(cutpoints) - 1:  # Ensure the last cutpoint is always included
                new_cutpoints.append(cutpoints[i])
                count_since_last_added = 0  # Reset count after adding a cutpoint

    return new_cutpoints


def drn_cutpoints(c_0, c_K, p, y, min_obs):
    cutpoints = uniform_cutpoints(c_0, c_K, p, y)
    return merge_cutpoints(cutpoints, y, min_obs)

 