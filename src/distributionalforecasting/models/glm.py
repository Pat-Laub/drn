from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class GLM(nn.Module):
    """
    A base PyTorch model representing a generalized linear model.
    This class is extended by specific distribution types like Gamma or Gaussian.
    """

    def __init__(self, p: int, distribution: str):
        """
        Args:
            p: the number of features in the model
            distribution: the type of GLM ('gamma' or 'gaussian')
        """
        super(GLM, self).__init__()
        self.p = p
        self.distribution = distribution
        self.linear = nn.Linear(p, 1)

        if distribution == 'gamma':
            self.log_mean = self.linear
            self.dispersion: Optional[float] = None  # Gamma distribution parameter
        elif distribution == 'gaussian':
            self.sigma: Optional[float] = None  # Standard deviation for Gaussian distribution
        else:
            raise ValueError(f"Unsupported model type: {self.distribution}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.distribution == 'gamma':
            out = torch.exp(self.linear(x)).squeeze(-1)
        elif self.distribution == 'gaussian':
            out = self.linear(x).squeeze(-1)
        else:
            raise ValueError(f"Unsupported model type: {self.distribution}")
        assert out.shape == torch.Size([x.shape[0]])
        return out

    def clone(self) -> "GLM":
        """
        Create an independent copy of the model.
        """
        glm = GLM(self.p, self.distribution)
        glm.distribution =self.distribution
        glm.load_state_dict(self.state_dict())

        if self.distribution == 'gamma':
            glm.dispersion = self.dispersion
           
        elif self.distribution == 'gaussian':
            glm.sigma = self.sigma
            
        return glm

    def distributions(self, x: torch.Tensor):
        """
        Create distributional forecasts for the given inputs, specific to the model type.
        """
        if self.distribution == 'gamma':
            return self._gamma_distributions(x)
        elif self.distribution == 'gaussian':
            return self._gaussian_distributions(x)
        else:
            raise ValueError(f"Unsupported model type: {self.distribution}")

    def _gamma_distributions(self, x: torch.Tensor):
        if self.dispersion is None:
            raise RuntimeError("Dispersion parameter has not been estimated yet.")
        means = self.forward(x)
        # Assuming the mean is the alpha parameter and dispersion is the beta parameter
        alphas, betas = gamma_convert_parameters(self.forward(x), self.dispersion)
        dists = torch.distributions.Gamma(alphas, betas)
        return dists

    def _gaussian_distributions(self, x: torch.Tensor):
        if self.sigma is None:
            raise RuntimeError("Standard deviation has not been estimated yet.")
        mu = self.forward(x)
        dists = torch.distributions.Normal(mu, self.sigma)
        return dists

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the predicted means for the given observations.
        """
        return self.forward(x)
    
    def icdf(self, x: Union[np.ndarray, torch.Tensor], p, l = None, u = None, max_iter = 1000, tolerance = 1e-7) -> torch.Tensor:
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
        num_observations = dists.cdf(torch.Tensor([1]).unsqueeze(-1)).shape[1]  # Dummy call to cdf to determine the batch size
        percentiles_tensor = torch.full((1, num_observations), fill_value=p, dtype=torch.float32)
        
        # Initialise matrices for the bounds
        lower_bounds = l if l is not None else torch.Tensor([0]) #self.cutpoints[0] - (self.cutpoints[-1]-self.cutpoints[0]) 
        upper_bounds = u if u is not None else torch.Tensor([200])  # Adjust max value as needed

        lower_bounds = lower_bounds.repeat(num_observations).reshape(1, num_observations)
        upper_bounds = upper_bounds.repeat(num_observations).reshape(1, num_observations)

        for _ in range(max_iter):
            mid_points = (lower_bounds + upper_bounds) / 2
            cdf_vals = dists.cdf(mid_points)
            # cdf_vals = torch.zeros(1, num_observations) 
            # for i in range(num_observations):
            #     cur_mid = mid_points[:, i]
            #     cdf_vals[:, i] = dists.cdf(cur_mid.unsqueeze(-1))[:, i]

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
    
    def quantiles(self, x: torch.Tensor, percentiles: list, l = None, u = None, max_iter = 1000, tolerance = 1e-7) -> torch.Tensor:
        """
        Calculate the quantile values for the given observations and percentiles (cumulative probabilities * 100).
        """
        if self.distribution == 'gamma':
            dists = self._gamma_distributions(x)
            quantiles = [self.icdf(x, torch.tensor(percentile/100), l, u, max_iter, tolerance) for percentile in percentiles]
        elif self.distribution == 'gaussian':
            dists = self._gaussian_distributions(x)
            quantiles = [self.icdf(x, torch.tensor(percentile/100), l, u, max_iter, tolerance) for percentile in percentiles]
        else:
            raise ValueError(f"Unsupported model type: {self.distribution}")
        return torch.stack(quantiles, dim=1)[0]

    def quantiles_old(self, x: torch.Tensor, percentiles: list, grid: torch.Tensor) -> torch.Tensor:
        # Get the CDF values for each instance and cutpoint
        cdf_values = self.distributions(x).cdf(grid).detach().numpy()
        quantile_levels =  torch.zeros((len(percentiles), x.shape[0]))

        # For each instance and each percentile, find the closest cutpoint index
        for i, percentile in enumerate(percentiles):
            quantile_value = percentile / 100
            abs_diff = np.abs(cdf_values - quantile_value)
            closest_idx = abs_diff.argmin(axis=0)
            quantile_levels[i, :] = torch.Tensor([grid[idx] for idx in closest_idx]).reshape(1, x.shape[0])

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
        p: the number of features
    """
    n = mu.shape[0]
    return (torch.sum((y - mu) ** 2 / mu**2) / (n - p)).item()


def gamma_convert_parameters(
    mu: torch.Tensor, phi: float
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

def gaussian_estimate_sigma(mu: torch.Tensor, y: torch.Tensor, p: int) -> float:
    """
    For a Gaussian GLM, the dispersion parameter is estimated using the method of moments.
    Args:
        mu: the predicted means for the Gaussian distributions (shape: (n, 1))
        y: the observed values (shape: (n, 1))
        p: the number of features
    """
    n = mu.shape[0]
    variance_estimate = torch.sum((y - mu) ** 2)/(n-1)
    return (torch.sqrt(variance_estimate)).item()

