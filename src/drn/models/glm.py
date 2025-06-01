from typing import Union

import numpy as np
import statsmodels.api as sm
import torch
import torch.nn as nn
from statsmodels.genmod.families import Gamma, Gaussian, InverseGaussian

from .base import BaseModel


class GLM(BaseModel):
    def __init__(
        self,
        p: int,
        distribution: str,
        learning_rate=1e-3,
        default: bool = False,
        null_model: bool = False,
    ):
        if distribution not in ("gamma", "gaussian", "inversegaussian", "lognormal"):
            raise ValueError(f"Unsupported model type: {distribution}")

        super(GLM, self).__init__()
        self.p = p
        self.distribution = distribution
        self.linear = nn.Linear(p, 1, bias=True)

        if default or null_model:
            # Set weights and bias to zero
            self.linear.weight.data.fill_(0.0)
            self.linear.bias.data.fill_(0.0)
            # Set default dispersion to 1 for numerical stability
            self.dispersion = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        else:
            # Dispersion not set until model is trained
            self.dispersion = nn.Parameter(torch.tensor(torch.nan), requires_grad=False)

        self.loss_fn = (
            gaussian_deviance_loss
            if distribution == "gaussian"
            else gamma_deviance_loss
        )
        self.learning_rate = learning_rate

    @staticmethod
    def from_statsmodels(
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        distribution: str,
        null_model: bool = False,
    ):
        p = X.shape[1]

        if isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor):
            device = X.device
            X = X.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
        else:
            device = None

        if distribution == "lognormal":
            y = np.log(y)
        # Create null model (intercept only)
        # if null_model:
        #     X = np.ones((X.shape[0], 1))  # Only intercept
        #     p = 1  # Single coefficient (intercept)

        # Choose the correct family
        if distribution == "gamma":
            family = Gamma(link=sm.families.links.Log())
        elif distribution in ["gaussian", "lognormal"]:
            family = Gaussian()
        elif distribution == "inversegaussian":
            family = InverseGaussian(link=sm.families.links.Log())

        # Fit the GLM model
        model = sm.GLM(y, sm.add_constant(X), family=family)
        results = model.fit()
        betas = results.params
        if not isinstance(results.params, np.ndarray):
            betas = np.asarray(betas)

        # Create PyTorch GLM instance
        torch_glm = GLM(p, distribution)
        torch_glm.linear.weight.data = (
            torch.tensor(betas[1:], dtype=torch.float32).unsqueeze(0)
            if not null_model
            else torch.zeros((1, p))
        )
        torch_glm.linear.bias.data = torch.tensor([betas[0]], dtype=torch.float32)

        # Set dispersion parameter
        if distribution == "gamma":
            disp = results.scale.item()
        elif distribution == "inversegaussian":
            disp = results.scale.item()
        elif distribution in ["gaussian", "lognormal"]:
            disp = (results.scale**0.5).item()

        torch_glm.dispersion = nn.Parameter(torch.tensor(disp), requires_grad=False)

        if device:
            torch_glm = torch_glm.to(device)

        return torch_glm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.distribution in ["gamma", "inversegaussian"]:
            return torch.exp(self.linear(x)).squeeze(-1)
        return self.linear(x).squeeze(-1)

    def clone(self) -> "GLM":
        """
        Create an independent copy of the model.
        """
        glm = GLM(self.p, self.distribution)
        glm.load_state_dict(self.state_dict())
        return glm

    def distributions(self, x: torch.Tensor):
        if torch.isnan(self.dispersion):
            raise RuntimeError("Dispersion parameter has not been estimated yet.")

        if self.distribution == "gamma":
            alphas, betas = gamma_convert_parameters(self.forward(x), self.dispersion)
            return torch.distributions.Gamma(alphas, betas)
        elif self.distribution == "inversegaussian":
            return InverseGaussianTorch(self.forward(x), self.dispersion)
        elif self.distribution == "lognormal":
            return torch.distributions.LogNormal(self.forward(x), self.dispersion)
        else:
            return torch.distributions.Normal(self.forward(x), self.dispersion)

    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(self.forward(X), y)

    def update_dispersion(self, X: torch.Tensor, y: torch.Tensor) -> None:
        disp = estimate_dispersion(self.distribution, self.forward(X), y, self.p)
        self.dispersion = nn.Parameter(torch.tensor(disp), requires_grad=False)

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the predicted means for the given observations.
        """
        return self.forward(x)

    def icdf(
        self,
        x: Union[np.ndarray, torch.Tensor],
        p,
        l=None,
        u=None,
        max_iter=1000,
        tolerance=1e-7,
    ) -> torch.Tensor:
        """
        Calculate the inverse CDF (quantiles) of the distribution for the given cumulative probability.

        Args:
            p: cumulative probability values at which to evaluate icdf
            l: lower bound for the quantile search
            u: upper bound for the quantile search
            max_iter: maximum number of iterations permitted for the quantile search
            tolerance: stopping criteria for the search (precision)

        Returns:
            A tensor of shape (1, batch_shape) containing the inverse CDF values.
        """

        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
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
        if self.distribution not in [
            "gamma",
            "gaussian",
            "lognormal",
            "inversegaussian",
        ]:
            raise ValueError(f"Unsupported model type: {self.distribution}")

        quantiles = [
            self.icdf(x, torch.tensor(percentile / 100), l, u, max_iter, tolerance)
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


def gamma_estimate_dispersion(mu: torch.Tensor, y: torch.Tensor, p: int) -> float:
    """
    For a gamma GLM, the dispersion parameter is estimated using the method of moments.
    Args:
        mu: the predicted means for the gamma distributions (shape: (n, 1))
        y: the observed values (shape: (n, 1))
        p: the number of features (not including the intercept)
    """
    n = mu.shape[0]
    dof = n - (p + 1)
    return (torch.sum((y - mu) ** 2 / mu**2) / dof).item()


def gamma_convert_parameters(
    mu: torch.Tensor, phi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Our models predict the mean of the gamma distribution, but we need the shape and rate parameters.
    This function converts the mean and dispersion parameter into the shape and rate parameters.
    Args:
        mu: the predicted means for the gamma distributions (shape: (n,))
        phi: the dispersion parameter
    Returns:
        alpha: the shape parameter (shape: (n,))
        beta: the rate parameter (shape: (n,))
    """
    beta = 1.0 / (mu * phi)
    alpha = (1.0 / phi) * torch.ones_like(beta)
    return alpha, beta


def gaussian_estimate_sigma(mu: torch.Tensor, y: torch.Tensor) -> float:
    """
    For a Gaussian GLM, the dispersion parameter is estimated using the method of moments.
    Args:
        mu: the predicted means for the Gaussian distributions (shape: (n, 1))
        y: the observed values (shape: (n, 1))
        p: the number of features
    """
    n = mu.shape[0]
    variance_estimate = torch.sum((y - mu) ** 2) / (n - 1)
    return (torch.sqrt(variance_estimate)).item()


def estimate_dispersion(distribution: str, mu: torch.Tensor, y: torch.Tensor, p: int):
    """
    Estimate the dispersion parameter for different distributions.

    Parameters:
    distribution (str): The type of distribution ("gamma", "gaussian", "inversegaussian").
    mu (torch.Tensor): The predicted mean values.
    y (torch.Tensor): The observed target values.
    p (int): The number of model parameters.

    Returns:
    torch.Tensor: The estimated dispersion parameter.
    """
    if distribution == "gamma":
        return gamma_estimate_dispersion(mu, y, p)
    elif distribution == "gaussian":
        return gaussian_estimate_sigma(mu, y)
    elif distribution == "inversegaussian":
        return inversegaussian_estimate_dispersion(mu, y, p)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")


def inversegaussian_estimate_dispersion(
    mu: torch.Tensor, y: torch.Tensor, p: int
) -> float:
    n = mu.shape[0]
    dof = n - (p + 1)
    return (torch.sum(((y - mu) ** 2) / (mu**3)) / dof).item()


class InverseGaussianTorch(torch.distributions.Distribution):
    def __init__(self, mean: torch.Tensor, dispersion: torch.Tensor):
        self.mu = mean
        self.dispersion = dispersion

    def log_prob(self, y: torch.Tensor) -> torch.Tensor:
        term1 = -0.5 * torch.log(2 * torch.pi * self.dispersion * y**3)
        term2 = -((y - self.mu) ** 2) / (2 * self.dispersion * y * self.mu**2)
        return term1 + term2

    @property
    def mean(self) -> torch.Tensor:
        return self.mu

    def cdf(self, y: torch.Tensor) -> torch.Tensor:
        lambda_ = 1.0 / self.dispersion

        sqrt_term = torch.sqrt(lambda_ / y)
        z1 = sqrt_term * (y / self.mu - 1.0)
        z2 = -sqrt_term * (y / self.mu + 1.0)

        standard_normal = torch.distributions.Normal(0.0, 1.0)
        term1 = standard_normal.cdf(z1)
        term2 = torch.exp(2.0 * lambda_ / self.mu) * standard_normal.cdf(z2)

        return term1 + term2
