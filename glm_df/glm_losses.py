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


def gamma_deviance_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Tweedie deviance loss for the gamma distribution.
    Args:
        y_pred: the predicted values (shape: (n,))
        y_true: the observed values (shape: (n,))
    Returns:
        the deviance loss (shape: (,))
    """
    loss = 2 * (y_true / y_pred - torch.log(y_true / y_pred) - 1)
    return torch.mean(loss)

def gaussian_deviance_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Normal deviance loss for the Gaussian distribution.
    Args:
        y_pred: the predicted values (shape: (n,))
        y_true: the observed values (shape: (n,))
    Returns:
        the deviance loss (shape: (,))
    """
    loss = (y_true - y_pred) ** 2
    return torch.mean(loss)
