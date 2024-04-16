from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .glm import GLM, gamma_convert_parameters


class CANN(nn.Module):
    """
    The Combined Actuarial Neural Network (CANN) model adaptable for both gamma and Gaussian GLMs.
    """

    def __init__(
        self, glm: GLM, num_hidden_layers=2, hidden_size=50, dropout_rate = 0.2, train_glm=False
    ):
        """
        Args:
            glm: the GLM to use as the backbone, can be either Gamma or Gaussian GLM
            num_hidden_layers: the number of hidden layers in the neural network
            hidden_size: the number of neurons in each hidden layer
            train_glm: whether to retrain the GLM or not
        """
        super(CANN, self).__init__()

        self.p = glm.p
        self.glm = glm.clone()
        self.train_glm = train_glm
        self.distribution = glm.distribution
        

        if self.distribution == 'gamma':
            self.dispersion: Optional[float] = None
        elif self.distribution == 'gaussian':
            self.sigma: Optional[float] = None
        else:
            raise ValueError(f"Unsupported GLM type: {self.distribution}")

        layers = [nn.Linear(self.p, hidden_size), nn.LeakyReLU(), nn.Dropout(dropout_rate)]
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))  # Add dropout after each LeakyReLU activation

        layers.append(nn.Linear(hidden_size, 1))
        self.nn_output_layer = nn.Sequential(*layers)

    def glm_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the output for the GLM component.
        Args:
            x: the input features (shape: (n, p))
        Returns:
            the output depending on the GLM type
        """
        # If we don't want to retrain the GLM, then we don't need to track gradients
        with torch.set_grad_enabled(self.train_glm):
            if self.distribution == 'gamma':
                return torch.exp(self.glm.log_mean(x)).squeeze(-1)
            elif self.distribution == 'gaussian':
                return self.glm.forward(x).squeeze(-1)
            else:
                raise ValueError(f"Unsupported GLM type: {self.distribution}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the predicted outputs for the distributions.
        Args:
            x: the input features (shape: (n, p))
        Returns:
            the predicted outputs (shape: (n,))
        """
        if self.distribution == 'gamma':
            out = torch.exp(self.glm_output(x) + self.nn_output_layer(x).squeeze(-1))
        elif self.distribution == 'gaussian':
            out = self.glm_output(x) + self.nn_output_layer(x).squeeze(-1)
        else:
            raise ValueError(f"Unsupported GLM type: {self.distribution}")
        assert out.shape == torch.Size([x.shape[0]])
        return out

    def distributions(self, x: torch.Tensor):
        """
        Create distributional forecasts for the given inputs, specific to the model type.
        """
        if self.distribution == 'gamma':
            if self.dispersion is None:
                raise RuntimeError("Dispersion parameter has not been estimated yet.")
            alphas, betas = gamma_convert_parameters(self.forward(x), self.dispersion)
            dists = torch.distributions.Gamma(alphas, betas)
        elif self.distribution == 'gaussian':
            if self.sigma is None:
                raise RuntimeError("Standard deviation has not been estimated yet.")
            mu = self.forward(x)
            dists = torch.distributions.Normal(mu, self.sigma)
        else:
            raise ValueError(f"Unsupported GLM type: {self.distribution}")
        assert dists.batch_shape == torch.Size([x.shape[0]])
        return dists

    def mean(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the predicted means for the given observations, specific to the model type.
        """
        return self.forward(torch.Tensor(x)).detach().numpy().squeeze()

    
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
            quantiles = [self.icdf(x, torch.tensor(percentile / 100.0),\
                 l, u, max_iter, tolerance) for percentile in percentiles]
        elif self.distribution == 'gaussian':
            quantiles = [self.icdf(x, torch.tensor(percentile / 100.0),\
                 l, u, max_iter, tolerance) for percentile in percentiles]
        else:
            raise ValueError(f"Unsupported model type: {self.distribution}")
        return torch.stack(quantiles, dim=1)[0]

