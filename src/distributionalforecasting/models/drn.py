import numpy as np
import torch
import torch.nn as nn

from ..distributions.extended_histogram import ExtendedHistogram
from .ddr import nll_loss

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


        layers = [nn.Linear(num_features, hidden_size), nn.LeakyReLU(), nn.Dropout(dropout_rate)]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
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

        clipped_log_adjustments = self.log_adjustments(x)
        log_drn_non_norm_prob_masses = torch.log(glm_prob_masses) + clipped_log_adjustments
        clipped_log_drn_non_norm_prob_masses = torch.clip(log_drn_non_norm_prob_masses, min = -75, max = 75)
        drn_non_norm_prob_masses = torch.exp(clipped_log_drn_non_norm_prob_masses)

        if DEBUG:
            assert drn_non_norm_prob_masses.shape == (x.shape[0], num_regions)
        
        # Standardising the probability masses
        sum_probs = torch.sum(drn_non_norm_prob_masses, axis=1, keepdim=True)
        drn_prob_masses = drn_non_norm_prob_masses / sum_probs

        if DEBUG:
            assert drn_prob_masses.shape == (x.shape[0], num_regions)

        if DEBUG:
            assert torch.allclose(
                torch.sum(drn_prob_masses, axis=1),
                torch.ones(x.shape[0], device=x.device),
            )
            #old = torch.min(drn_prob_masses/torch.diff(self.cutpoints))
            if torch.min(drn_prob_masses/torch.diff(self.cutpoints)) == 0:
                drn_prob_masses += 1e-25
                sum_probs = torch.sum(drn_prob_masses, axis=1, keepdim=True)
                drn_prob_masses = drn_prob_masses / sum_probs
                
            assert torch.allclose(
                torch.sum(drn_prob_masses, axis=1),
                torch.ones(x.shape[0], device=x.device),
            )
            assert torch.min(drn_prob_masses/torch.diff(self.cutpoints)) > 0
            

        return baseline_dists, self.cutpoints, drn_prob_masses

    def distributions(self, x):
        
        baseline_dists, cutpoints, prob_masses = self.forward(x)
        return ExtendedHistogram(baseline_dists, cutpoints, prob_masses)


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

def uniform_cutpoints(c_0, c_K, p, y):
    num_cutpoints = int(np.ceil(p * len(y)))
    return list(np.linspace(c_0, c_K, num_cutpoints))
    
def merge_cutpoints(cutpoints: list[float], y: np.ndarray, min_obs: int) -> list[float]:
    # Ensure cutpoints are sorted and unique to start with
    cutpoints = sorted(list(np.unique(cutpoints)))
    assert len(cutpoints) >= 2

    new_cutpoints = [cutpoints[0]]  # Start with the first cutpoint
    left = 0
    
    for right in range(1, len(cutpoints) - 1):
        num_in_region = np.sum((y >= cutpoints[left]) & (y < cutpoints[right]))
        num_after_region = np.sum((y >= cutpoints[right]) & (y < cutpoints[-1]))

        if num_in_region >= min_obs and num_after_region >= min_obs:
            new_cutpoints.append(cutpoints[right])
            left = right

    new_cutpoints.append(cutpoints[-1])  # End with the last cutpoint

    return new_cutpoints


def drn_cutpoints(c_0, c_K, p, y, min_obs):
    cutpoints = uniform_cutpoints(c_0, c_K, p, y)
    return merge_cutpoints(cutpoints, y, min_obs)
