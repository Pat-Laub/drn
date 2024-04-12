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

from .extended_histogram_class import ExtendedHistogram
from .histogram_class import Histogram

def jbce_loss(dists, y, alpha=0.0):
    """
    The joint binary cross entropy loss.
    Args:
        dists: the predicted distributions
        y: the observed values
        alpha: the penalty parameter 
    """
    
    cutpoints = dists.cutpoints
    cdf_at_cutpoints = dists.cdf_at_cutpoints()

    assert cdf_at_cutpoints.shape == torch.Size([len(cutpoints), len(y)])

    n = y.shape[0]
    C = len(cutpoints)

    # The cross entropy loss can't accept 0s or 1s for the cumulative probabilities.
    epsilon = 1e-15
    cdf_at_cutpoints = cdf_at_cutpoints.clamp(epsilon, 1 - epsilon)

    # Change: C to C-1
    losses = torch.zeros(C - 1, n, device=y.device, dtype=y.dtype)

    for i in range(1, C):
        targets = (y <= cutpoints[i]).float()
        probs = cdf_at_cutpoints[i, :]
        losses[i - 1, :] = nn.functional.binary_cross_entropy(
            probs, targets, reduction="none"
        )

    return torch.mean(losses)
    
def drn_nll_loss(pred, y,
                    kl_alpha = 0,
                    mean_alpha = 0,
                    tv_alpha = 0,
                    dv_alpha = 0):
    
    baseline_dists, cutpoints, prob_masses = pred
    dists = ExtendedHistogram(baseline_dists, cutpoints, prob_masses)
    losses = nll_loss(dists, y)
    baseline_probs = torch.diff(baseline_dists.cdf(cutpoints.unsqueeze(-1)), axis = 0).T
    if tv_alpha > 0 or dv_alpha > 0:
        drn_density = prob_masses/torch.diff(cutpoints)
        first_order_diffs = torch.diff(drn_density, dim=1)
        
    if kl_alpha > 0:
        epsilon = 1e-12
        hist_weight = baseline_dists.cdf(cutpoints[-1]) - baseline_dists.cdf(cutpoints[0])
        real_adjustments = (prob_masses) / (baseline_probs + epsilon) * hist_weight.view(-1, 1)
        penalty_loss = baseline_probs * real_adjustments * torch.log(real_adjustments+ epsilon) 
        losses += torch.mean(torch.sum(penalty_loss, axis = 0)) * kl_alpha
        
    if mean_alpha > 0:
        means_glm = baseline_dists.mean
        means_drn = dists.mean()
        mean_change = (means_drn - means_glm) 
        losses += torch.mean(mean_change ** 2) * mean_alpha

    if tv_alpha > 0:
        losses += torch.mean(torch.sum(torch.abs(first_order_diffs), dim = 1)) * tv_alpha

    if dv_alpha > 0:
        second_order_diffs = torch.diff(first_order_diffs, dim=1)
        losses += torch.mean(torch.sum(second_order_diffs**2, dim=1)) * dv_alpha

    return losses
    
 
def drn_jbce_loss(pred, y,
                    kl_alpha = 0,
                    mean_alpha = 0,
                    tv_alpha = 0,
                    dv_alpha = 0):
    
    baseline_dists, cutpoints, prob_masses = pred
    dists = ExtendedHistogram(baseline_dists, cutpoints, prob_masses)
    losses = jbce_loss(dists, y)
    baseline_probs = torch.diff(baseline_dists.cdf(cutpoints.unsqueeze(-1)), axis = 0).T
    if tv_alpha > 0 or dv_alpha > 0:
        drn_density = prob_masses/torch.diff(cutpoints)
        first_order_diffs = torch.diff(drn_density, dim=1)
        
    if kl_alpha > 0:
        epsilon = 1e-12
        hist_weight = baseline_dists.cdf(cutpoints[-1]) - baseline_dists.cdf(cutpoints[0])
        real_adjustments = (prob_masses) / (baseline_probs + epsilon) * hist_weight.view(-1, 1)
        penalty_loss = baseline_probs * real_adjustments * torch.log(real_adjustments+ epsilon) 
        losses += torch.mean(torch.sum(penalty_loss, axis = 0)) * kl_alpha
        
    if mean_alpha > 0:
        means_glm = baseline_dists.mean
        means_drn = dists.mean()
        mean_change = (means_drn - means_glm) 
        losses += torch.mean(mean_change ** 2) * mean_alpha

    if tv_alpha > 0:
        losses += torch.mean(torch.sum(torch.abs(first_order_diffs), dim = 1)) * tv_alpha

    if dv_alpha > 0:
        second_order_diffs = torch.diff(first_order_diffs, dim=1)
        losses += torch.mean(torch.sum(second_order_diffs**2, dim=1)) * dv_alpha

    return losses