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





def gamma_mdn_loss(out: list[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    """
    Calculate the negative log-likelihood loss for the mixture density network.
    Args:
        out: the mixture weights, shape parameters and rate parameters (all shape: (n, num_components))
        y: the observed values (shape: (n, 1))
    Returns:
        the negative log-likelihood loss (shape: (1,))
    """
    weights, alphas, betas = out
    dists = MixtureSameFamily(
        Categorical(weights),
        torch.distributions.Gamma(alphas, betas),
    )
    log_prob = dists.log_prob(y.squeeze())
    assert log_prob.ndim == 1
    return -log_prob.mean()


def gaussian_mdn_loss(out: list[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    """
    Calculate the negative log-likelihood loss for the mixture density network.
    Args:
        out: the mixture weights, shape parameters and rate parameters (all shape: (n, num_components))
        y: the observed values (shape: (n, 1))
    Returns:
        the negative log-likelihood loss (shape: (1,))
    """
    weights, mus, sigmas = out
    dists = MixtureSameFamily(
        Categorical(weights),
        torch.distributions.Normal(mus, sigmas),
    )
    log_prob = dists.log_prob(y.squeeze())
    assert log_prob.ndim == 1
    return -log_prob.mean()
