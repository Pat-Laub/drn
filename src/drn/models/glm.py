from __future__ import annotations
from typing import Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
from statsmodels.genmod.families import Gamma, Gaussian, InverseGaussian

from .base import BaseModel
from ..distributions import inverse_gaussian
from ..distributions.estimation import gamma_convert_parameters, estimate_dispersion
from ..utils import _to_numpy


class GLM(BaseModel):
    def __init__(self, distribution: str, p: Optional[int] = None, learning_rate=1e-3):
        self.save_hyperparameters()

        if distribution not in ("gamma", "gaussian", "inversegaussian", "lognormal"):
            raise ValueError(f"Unsupported model type: {distribution}")

        super(GLM, self).__init__()
        self.distribution = distribution

        # Set default dispersion to 1 for numerical stability
        self.dispersion = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)

        self.loss_fn = (
            gaussian_deviance_loss
            if distribution == "gaussian"
            else gamma_deviance_loss
        )

        self.learning_rate = learning_rate
        self.p = p
        if self.p is not None:
            self._initialise_weights()

    def _initialise_weights(self):
        self.linear = nn.Linear(self.p, 1, bias=True)

        # Set weights and bias to zero
        self.linear.weight.data.fill_(0.0)
        self.linear.bias.data.fill_(0.0)

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.DataFrame, pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        grad_descent: bool = False,
        *args,
        **kwargs,
    ) -> GLM:
        if self.p is None:
            self.p = X_train.shape[1]
            self._initialise_weights()

        # If the user specifically wants to use gradient descent, we will use the base class fit method
        if grad_descent:
            super().fit(X_train, y_train, *args, **kwargs)
            self.update_dispersion(X_train, y_train)
            return self

        # But by default, fit using statsmodels and copy the parameters
        fitted_glm = self.from_statsmodels(X_train, y_train, self.distribution)
        self.linear.weight.data = fitted_glm.linear.weight.data.clone()
        self.linear.bias.data = fitted_glm.linear.bias.data.clone()
        self.dispersion = nn.Parameter(
            fitted_glm.dispersion.data.clone(), requires_grad=False
        )
        return self

    @staticmethod
    def from_statsmodels(
        X: Union[np.ndarray, torch.Tensor, pd.DataFrame, pd.Series],
        y: Union[np.ndarray, torch.Tensor, pd.DataFrame, pd.Series],
        distribution: str,
        null_model: bool = False,
    ):
        # Convert inputs to numpy arrays with float32 precision
        X = _to_numpy(X)
        y = _to_numpy(y).flatten()

        p = X.shape[1]

        if distribution == "lognormal":
            y = np.log(y)

        # Choose the correct family
        if distribution == "gamma":
            family = Gamma(link=sm.families.links.Log())
        elif distribution in ["gaussian", "lognormal"]:
            family = Gaussian()
        elif distribution == "inversegaussian":
            family = InverseGaussian(link=sm.families.links.Log())

        # Fit the GLM model
        if null_model:
            # Just input the constant without covariates
            ones = np.ones((X.shape[0], 1))
            model = sm.GLM(y, ones, family=family)
        else:
            model = sm.GLM(y, sm.add_constant(X), family=family)

        results = model.fit()
        betas = results.params
        if not isinstance(results.params, np.ndarray):
            betas = np.asarray(betas)

        # Create PyTorch GLM instance
        torch_glm = GLM(distribution, p=p)
        torch_glm.linear.weight.data = (
            torch.Tensor(betas[1:]).unsqueeze(0)
            if not null_model
            else torch.zeros((1, p))
        )
        torch_glm.linear.bias.data = torch.Tensor([betas[0]])

        # Set dispersion parameter
        if distribution == "gamma":
            disp = results.scale.item()
        elif distribution == "inversegaussian":
            disp = results.scale.item()
        elif distribution in ["gaussian", "lognormal"]:
            disp = (results.scale**0.5).item()

        torch_glm.dispersion = nn.Parameter(torch.Tensor([disp]), requires_grad=False)

        return torch_glm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.distribution in ["gamma", "inversegaussian"]:
            return torch.exp(self.linear(x)).squeeze(-1)
        return self.linear(x).squeeze(-1)

    def clone(self) -> GLM:
        """
        Create an independent copy of the model.
        """
        glm = GLM(self.distribution, p=self.p)
        glm.load_state_dict(self.state_dict())
        return glm

    def distributions(
        self, x: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor]
    ):
        if torch.isnan(self.dispersion):
            raise RuntimeError("Dispersion parameter has not been estimated yet.")

        x = self._to_tensor(x)
        if self.distribution == "gamma":
            alphas, betas = gamma_convert_parameters(self.forward(x), self.dispersion)
            return torch.distributions.Gamma(alphas, betas)
        elif self.distribution == "inversegaussian":
            return inverse_gaussian.InverseGaussian(self.forward(x), self.dispersion)
        elif self.distribution == "lognormal":
            return torch.distributions.LogNormal(self.forward(x), self.dispersion)
        else:
            return torch.distributions.Normal(self.forward(x), self.dispersion)

    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(self.forward(X), y)

    def update_dispersion(
        self,
        X_train: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y_train: Union[np.ndarray, pd.Series, torch.Tensor],
    ) -> None:
        X = self._to_tensor(X_train)
        y = self._to_tensor(y_train)
        disp = estimate_dispersion(self.distribution, self.forward(X), y, self.p)
        self.dispersion = nn.Parameter(torch.Tensor([disp]), requires_grad=False)

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the predicted means for the given observations.
        """
        return self.forward(x)

    def icdf(
        self,
        x: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor],
        p,
        l=None,
        u=None,
        max_iter=1000,
        tolerance=1e-7,
    ) -> torch.Tensor:
        """
        Calculate the inverse CDF (quantiles) of the distribution for the given cumulative probability.

        Args:
            x: Input features
            p: cumulative probability values at which to evaluate icdf
            l: lower bound for the quantile search
            u: upper bound for the quantile search
            max_iter: maximum number of iterations permitted for the quantile search
            tolerance: stopping criteria for the search (precision)

        Returns:
            A tensor of shape (1, batch_shape) containing the inverse CDF values.
        """
        x = self._to_tensor(x)
        dists = self.distributions(x)
        num_observations = dists.cdf(torch.Tensor([1]).unsqueeze(-1)).shape[
            1
        ]  # Dummy call to cdf to determine the batch size
        percentiles_tensor = torch.full(
            (1, num_observations), fill_value=p, dtype=torch.float32
        )

        # Initialise matrices for the bounds
        lower_bounds = (
            l if l is not None else torch.Tensor([0])
        )  # self.cutpoints[0] - (self.cutpoints[-1]-self.cutpoints[0])
        upper_bounds = (
            u if u is not None else torch.Tensor([200])
        )  # Adjust max value as needed

        lower_bounds = lower_bounds.repeat(num_observations).reshape(
            1, num_observations
        )
        upper_bounds = upper_bounds.repeat(num_observations).reshape(
            1, num_observations
        )

        for _ in range(max_iter):
            mid_points = (lower_bounds + upper_bounds) / 2
            cdf_vals = dists.cdf(mid_points)

            # Update the bounds based on where the CDF values are relative to the target percentiles
            lower_update = cdf_vals < percentiles_tensor
            upper_update = ~lower_update
            lower_bounds = torch.where(lower_update, mid_points, lower_bounds)
            upper_bounds = torch.where(upper_update, mid_points, upper_bounds)

            # Check for convergence
            if torch.max(upper_bounds - lower_bounds) < tolerance:
                break

        # Use the midpoint between the final bounds as the quantile estimate
        quantiles = (lower_bounds + upper_bounds) / 2

        return quantiles

    def quantiles(
        self,
        x: torch.Tensor,
        percentiles: list,
        l=None,
        u=None,
        max_iter=1000,
        tolerance=1e-7,
    ) -> torch.Tensor:
        """
        Calculate the quantile values for the given observations and percentiles (cumulative probabilities * 100).
        """
        quantiles = [
            self.icdf(x, percentile / 100, l, u, max_iter, tolerance)
            for percentile in percentiles
        ]
        return torch.stack(quantiles, dim=1)[0]

    def quantiles_old(
        self, x: torch.Tensor, percentiles: list, grid: torch.Tensor
    ) -> torch.Tensor:
        # Get the CDF values for each instance and cutpoint
        cdf_values = self.distributions(x).cdf(grid).detach().numpy()
        quantile_levels = torch.zeros((len(percentiles), x.shape[0]))

        # For each instance and each percentile, find the closest cutpoint index
        for i, percentile in enumerate(percentiles):
            quantile_value = percentile / 100
            abs_diff = np.abs(cdf_values - quantile_value)
            closest_idx = abs_diff.argmin(axis=0)
            quantile_levels[i, :] = torch.Tensor(
                [grid[idx] for idx in closest_idx]
            ).reshape(1, x.shape[0])

        return quantile_levels


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
