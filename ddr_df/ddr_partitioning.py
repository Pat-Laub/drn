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


def ddr_cutpoints(c_0, c_K, p, y):
    num_cutpoints = int(np.ceil(p * len(y)))
    cutpoints = list(np.linspace(c_0, c_K, num_cutpoints))

    return(cutpoints)