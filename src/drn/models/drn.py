import numpy as np
import torch
import torch.nn as nn

from ..distributions.extended_histogram import ExtendedHistogram
from .ddr import nll_loss


class DRN(nn.Module):
    def __init__(
        self,
        num_features,
        cutpoints,
        glm,
        num_hidden_layers=2,
        hidden_size=75,
        dropout_rate=0.2,
        baseline_start=False,
    ):
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
        self.glm = glm.clone() if hasattr(glm, "clone") else glm

        layers = [
            nn.Linear(num_features, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        ]
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

            baseline_cdfs = baseline_dists.cdf(self.cutpoints.unsqueeze(-1)).T
            if DEBUG:
                assert baseline_cdfs.shape == (x.shape[0], num_cutpoints)

            baseline_probs = torch.diff(baseline_cdfs, dim=1)
            if DEBUG:
                assert baseline_probs.shape == (x.shape[0], num_regions)

            # Sometimes the GLM probabilities are 0 simply due to numerical problems.
            # DRN cannot adjust regions with 0 probability, so we ensure 0's become
            # an incredibly small number just to avoid this issue.
            mass = torch.sum(baseline_probs, axis=1, keepdim=True)
            baseline_probs = torch.clip(baseline_probs, min=1e-10, max=1.0)
            baseline_probs = (
                baseline_probs / torch.sum(baseline_probs, axis=1, keepdim=True) * mass
            )

        drn_logits = torch.log(baseline_probs) + self.log_adjustments(x)
        drn_pmf = torch.softmax(drn_logits, dim=1)

        if DEBUG:
            assert drn_pmf.shape == (x.shape[0], num_regions)

            # Sometimes we get nan value in here. Otherwise, it should sum to 1.
            assert torch.isnan(drn_pmf).any() or torch.allclose(
                torch.sum(drn_pmf, axis=1),
                torch.ones(x.shape[0], device=x.device),
            )

        return baseline_dists, self.cutpoints, baseline_probs, drn_pmf

    def distributions(self, x):
        baseline_dists, cutpoints, baseline_probs, drn_pmf = self.forward(x)
        return ExtendedHistogram(baseline_dists, cutpoints, drn_pmf, baseline_probs)


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


def drn_loss(
    pred,
    y,
    kind="jbce",
    kl_alpha=0,
    mean_alpha=0,
    tv_alpha=0,
    dv_alpha=0,
    kl_direction="forwards",
):
    baseline_dists, cutpoints, baseline_probs, drn_pmf = pred
    dists = ExtendedHistogram(baseline_dists, cutpoints, drn_pmf, baseline_probs)

    if kind == "jbce":
        losses = jbce_loss(dists, y)
    else:
        losses = nll_loss(dists, y)

    a_i = dists.real_adjustments()
    b_i = baseline_probs

    if kl_alpha > 0:
        epsilon = 1e-30
        if kl_direction == "forwards":
            kl = -(torch.log(a_i + epsilon) * b_i)
        else:
            kl = torch.log(a_i + epsilon) * a_i * b_i
        losses += torch.mean(torch.sum(kl, axis=0)) * kl_alpha

    if mean_alpha > 0:
        losses += torch.mean((baseline_dists.mean - dists.mean) ** 2) * mean_alpha

    if tv_alpha > 0 or dv_alpha > 0:
        drn_density = a_i * b_i / torch.diff(cutpoints)
        first_diffs = torch.diff(drn_density, dim=1)

        if tv_alpha > 0:
            losses += torch.mean(torch.sum(torch.abs(first_diffs), dim=1)) * tv_alpha

        if dv_alpha > 0:
            second_diffs = torch.diff(first_diffs, dim=1)
            losses += torch.mean(torch.sum(second_diffs**2, dim=1)) * dv_alpha

    return losses


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


def drn_cutpoints(c_0, c_K, y, proportion=None, num_cutpoints=None, min_obs=1):
    if proportion is None and num_cutpoints is None:
        raise ValueError(
            "Either a proportion p or a specific num_cutpoints must be provided."
        )

    if proportion is not None:
        num_cutpoints = int(np.ceil(proportion * len(y)))

    uniform_cutpoints = list(np.linspace(c_0, c_K, num_cutpoints))

    return merge_cutpoints(uniform_cutpoints, y, min_obs)
