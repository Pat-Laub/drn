import torch
import torch.nn as nn
from torch.distributions import Distribution

from .histogram import Histogram


class ExtendedHistogram(Distribution):
    """
    This class represents a splicing of a supplied distribution with a histogram distribution.
    The histogram part is defined by K regions with boundaries -infty < c_0 < c_1 < ... < c_K < infty.
    The final density before c_0 & after c_K is the same as the original distribution.
    The density between c_k & c_{k+1} is defined by the histogram distribution.
    """

    def __init__(
        self,
        baseline: Distribution,
        cutpoints: torch.Tensor,
        prob_masses: torch.Tensor,
    ):
        """
        Args:
            baseline: the original distribution
            regions: the bin boundaries (shape: (K+1,))
            prob_masses: the probability for landing in each regions (shape: (n, K))
        """
        self.baseline = baseline
        self.prob_masses = prob_masses
        self.cutpoints = cutpoints
        self.histogram = Histogram(cutpoints, prob_masses)
        self.scale_down_hist =  (
            baseline.cdf(cutpoints[-1]) - baseline.cdf(cutpoints[0])
        )
        
        assert self.scale_down_hist.shape == torch.Size([self.histogram.batch_shape[0]])

        super(ExtendedHistogram, self).__init__(
            batch_shape=self.histogram.batch_shape, validate_args=False
        )

    def baseline_prob_between_cutpoints(self) -> torch.Tensor:
        """
        Calculate the baseline probability vector
        """
        return(self.baseline_prob_masses)

    def real_adjustments(self) -> torch.Tensor:
        """
        Calculate the real adjustment factors a_k's
        """
        return(self.real_adjustments)

    def prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Calculate the probability densities of `values`.
        """
        
        orig_ndim = value.ndim
        
        # Ensure the last dimension of value matches the batch_shape
        if value.shape[-1] != self.batch_shape[0]:
            if value.ndim == 1:
                value = value.unsqueeze(-1)
            value = value.expand(-1, self.batch_shape[0])

        # Ensure value is 2D
        if value.ndim == 1:
            value = value.unsqueeze(0)

        baseline_prob = torch.exp(self.baseline.log_prob(value))
        hist_prob = self.histogram.prob(value) * (self.scale_down_hist + 1e-10)

        in_hist = (value >= self.histogram.cutpoints[0]) & (
            value < self.histogram.cutpoints[-1]
        )
        in_baseline = ~in_hist

        probabilities = torch.zeros_like(baseline_prob)
        probabilities[in_baseline] = baseline_prob[in_baseline]
        probabilities[in_hist] = hist_prob[in_hist]

        return probabilities

    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return torch.log(self.prob(value))

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        """
        Calculate the cumulative distribution function for the given values.
        """
        baseline_cdf = self.baseline.cdf(value)
        hist_cdf = self.histogram.cdf(value) * self.scale_down_hist
        in_hist = (value >= self.histogram.cutpoints[0]) & (
            value < self.histogram.cutpoints[-1]
        )
        in_hist = in_hist.expand(value.shape[0], self.batch_shape[0]) if in_hist.ndim > 1 else in_hist
        in_baseline = ~in_hist

        lower_cdf = self.baseline.cdf(self.histogram.cutpoints[0])
        cdf_values = torch.zeros_like(baseline_cdf)

        cdf_values[in_baseline] = baseline_cdf[in_baseline]
        cdf_values[in_hist] = (lower_cdf + hist_cdf)[in_hist]

        return cdf_values


    def cdf_at_cutpoints(self) -> torch.Tensor:
        """
        Calculate the cumulative distribution function at each cutpoint.
        """
        hist_at_cutpoints = (
            self.histogram.cdf_at_cutpoints() * self.scale_down_hist.unsqueeze(0)
        )
        lower_cdf = self.baseline.cdf(self.histogram.cutpoints[0]).unsqueeze(0)
        out = lower_cdf + hist_at_cutpoints
        return out

    @property    
    def mean(self) -> torch.Tensor:
        """
        Calculate the mean of the distribution.
        Returns:
            the mean (shape: (batch_shape,))
        """
        middle_of_bins = (self.cutpoints[1:] + self.cutpoints[:-1]) / 2
        return torch.sum(self.prob_masses * middle_of_bins, dim=1)
    
    
    def icdf(self, p, l = None, u = None, max_iter = 1000, tolerance = 1e-7) -> torch.Tensor:
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

        num_observations = self.cdf(torch.Tensor([1]).unsqueeze(-1)).shape[1]  # Dummy call to cdf to determine the batch size
        percentiles_tensor = torch.full((1, num_observations), fill_value=p, dtype=torch.float32)

        
       
        # Initialise matrices for the bounds
        lower_bounds = l if l is not None else torch.Tensor([0]) #self.cutpoints[0] - (self.cutpoints[-1]-self.cutpoints[0]) 
        upper_bounds = u if u is not None else self.cutpoints[-1] + (self.cutpoints[-1]-self.cutpoints[0])  # Adjust max value as needed

        lower_bounds = lower_bounds.repeat(num_observations).reshape(1, num_observations)
        upper_bounds = upper_bounds.repeat(num_observations).reshape(1, num_observations)
        
        for _ in range(max_iter):
            mid_points = (lower_bounds + upper_bounds) / 2

            cdf_vals = self.cdf(mid_points)

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
    
    
    def quantiles(self, percentiles: list, l = None, u = None, max_iter = 1000, tolerance = 1e-7) -> torch.Tensor:
        """
        Calculate the quantile values for the given observations and percentiles (cumulative probabilities * 100).
        """
        l = torch.Tensor([0]) #self.cutpoints[0] - (self.cutpoints[-1]-self.cutpoints[0]) if l is None else l
        u = self.cutpoints[-1] + (self.cutpoints[-1]-self.cutpoints[0]) if u is None else u
        quantiles = [self.icdf(torch.tensor(percentile / 100.0), l, u, max_iter, tolerance) for percentile in percentiles]

        return torch.stack(quantiles, dim=1)[0]
