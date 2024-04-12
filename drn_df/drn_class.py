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

from .histogram_class import Histogram
from .extended_histogram_class import ExtendedHistogram

class DRN(nn.Module):
    def __init__(self, num_features, cutpoints, glm, num_hidden_layers=2, hidden_size=75, dropout_rate = 0.2,
                baseline_start = False):
        """
        Args:
            num_features: Number of features in the input dataset.
            cutpoints: Cutpoints for the DRN model.
            glm: A Generalized Linear Model (GLM) that DRN will adjust.
            num_hidden_layers: Number of hidden layers in the DRN network.
            hidden_size: Number of neurons in each hidden layer.
        """
        super(DRN, self).__init__()
        self.cutpoints = nn.Parameter(torch.Tensor(cutpoints), requires_grad=False)
        # Assuming glm.clone() is a method to clone the glm model; ensure glm has a clone method.
        self.glm = glm.clone() if hasattr(glm, 'clone') else glm
        self.num_hidden_layers = num_hidden_layers


        layers = [nn.Linear(num_features, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate)) 
            
        self.hidden_layers = nn.Sequential(*layers)

        # Output layer
        self.fc_output = nn.Linear(hidden_size, len(self.cutpoints) - 1)
        self.batch_norm = nn.BatchNorm1d(len(cutpoints) - 1)

        # Initialize weights and biases for fc_output to zero
        if baseline_start:
            nn.init.constant_(self.fc_output.weight, 0)
            nn.init.constant_(self.fc_output.bias, 0)
        

    def log_adjustments(self, x):
        """
        Estimates log adjustments using the neural network.
        Args:
            x: Input features.
        Returns:
            Log adjustments for the DRN model.
        """
        # Pass input through the hidden layers
        z = self.hidden_layers(x)
        # Compute log adjustments
        log_adjustments = self.fc_output(z)
        return log_adjustments
    
        # normalized_log_adjustments = self.batch_norm(log_adjustments)
        # return normalized_log_adjustments
    
    def forward(self, x):
        DEBUG = True
        if DEBUG:
            num_cutpoints = len(self.cutpoints)
            num_regions = len(self.cutpoints) - 1

        with torch.no_grad():
            baseline_dists = self.glm.distributions(x)

            glm_cdfs = baseline_dists.cdf(self.cutpoints.unsqueeze(-1)).T

            if DEBUG:
                assert glm_cdfs.shape == (x.shape[0], num_cutpoints)

            glm_prob_masses = torch.diff(glm_cdfs, dim=1)
            if DEBUG:
                assert glm_prob_masses.shape == (x.shape[0], num_regions)

        clipped_log_adjustments = torch.clamp(self.log_adjustments(x), min=-75, max=75)  
        adjustments = torch.exp(clipped_log_adjustments)

        # adjustments = torch.exp(self.log_adjustments(x))
        # adjustments = adjustments.clamp(max=1e10, min=1e-10)

        # Multiplying by the adjustment factors
        addb_non_norm_prob_masses = (glm_prob_masses + 1e-10) * (adjustments)

        if DEBUG:
            assert addb_non_norm_prob_masses.shape == (x.shape[0], num_regions)
        
        # Standardising the probability masses
        sum_probs = torch.sum(addb_non_norm_prob_masses, axis=1, keepdim=True)
        addb_prob_masses = addb_non_norm_prob_masses / sum_probs

        if DEBUG:
            assert addb_prob_masses.shape == (x.shape[0], num_regions)

        if DEBUG:
            assert torch.allclose(
                torch.sum(addb_prob_masses, axis=1),
                torch.ones(x.shape[0], device=x.device),
            )
            #old = torch.min(addb_prob_masses/torch.diff(self.cutpoints))
            if torch.min(addb_prob_masses/torch.diff(self.cutpoints)) == 0:
                addb_prob_masses += 1e-25
                sum_probs = torch.sum(addb_prob_masses, axis=1, keepdim=True)
                addb_prob_masses = addb_prob_masses / sum_probs
                
            assert torch.allclose(
                torch.sum(addb_prob_masses, axis=1),
                torch.ones(x.shape[0], device=x.device),
            )
            assert torch.min(addb_prob_masses/torch.diff(self.cutpoints)) > 0
            

        return baseline_dists, self.cutpoints, addb_prob_masses

    def distributions(self, x):
        
        baseline_dists, cutpoints, prob_masses = self.forward(x)
        return ExtendedHistogram(baseline_dists, cutpoints, prob_masses)