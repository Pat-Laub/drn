from __future__ import annotations
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .glm import (
    GLM,
    gamma_convert_parameters,
    estimate_dispersion,
    gamma_deviance_loss,
    gaussian_deviance_loss,
)

from .base import BaseModel
from .constant import Constant


class CANN(BaseModel):
    """
    The Combined Actuarial Neural Network (CANN) model adaptable for both gamma and Gaussian GLMs.
    """

    def __init__(
        self,
        baseline: Union[GLM, Constant],
        num_hidden_layers=2,
        hidden_size=50,
        dropout_rate=0.2,
        train_glm=False,
        learning_rate=1e-3,
    ):
        """
        Args:
            baseline_model: the baseline model to use (GLM or Constant)
            num_hidden_layers: the number of hidden layers in the neural network
            hidden_size: the number of neurons in each hidden layer
            train_glm: whether to retrain the baseline model or not
        """
        self.save_hyperparameters()

        if not baseline.distribution in ("gamma", "gaussian", "inversegaussian"):
            raise ValueError(f"Unsupported model type: {baseline.distribution}")

        super(CANN, self).__init__()

        # Store baseline model and lazy initialization parameters
        self.baseline = baseline.clone()
        self.train_glm = train_glm
        self.distribution = baseline.distribution
        self.dispersion = nn.Parameter(torch.Tensor([torch.nan]), requires_grad=False)

        layers = [nn.LazyLinear(hidden_size), nn.LeakyReLU(), nn.Dropout(dropout_rate)]
        for _ in range(num_hidden_layers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_size, hidden_size),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
        layers.append(nn.Linear(hidden_size, 1))

        self.nn_output_layer = nn.Sequential(*layers)

        self.loss_fn = (
            gaussian_deviance_loss
            if self.distribution == "gaussian"
            else gamma_deviance_loss
        )
        self.learning_rate = learning_rate

    def fit(self, X_train, y_train, *args, **kwargs) -> CANN:
        super().fit(X_train, y_train, *args, **kwargs)
        self.update_dispersion(X_train, y_train)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the predicted outputs for the distributions.
        Args:
            x: the input features (shape: (n, p))
        Returns:
            the predicted outputs (shape: (n,))
        """
        if self.distribution in ["gamma", "inversegaussian"]:
            out = torch.exp(
                torch.log(self.baseline.forward(x)) + self.nn_output_layer(x).squeeze()
            )
        else:
            out = self.baseline.forward(x) + self.nn_output_layer(x).squeeze()

        assert out.shape == torch.Size(
            [x.shape[0]]
        ), f"Expected output shape [n], got {out.shape} for input shape {x.shape}"
        return out

    def distributions(
        self, x: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor]
    ) -> Union[torch.distributions.Gamma, torch.distributions.Normal]:
        """
        Create distributional forecasts for the given inputs, specific to the model type.
        """
        if torch.isnan(self.dispersion):
            raise RuntimeError("Dispersion parameter has not been estimated yet.")

        x = self._to_tensor(x)
        if self.distribution == "gamma":
            alphas, betas = gamma_convert_parameters(self.forward(x), self.dispersion)
            dists = torch.distributions.Gamma(alphas, betas)
        else:
            dists = torch.distributions.Normal(self.forward(x), self.dispersion)

        assert dists.batch_shape == torch.Size([x.shape[0]])
        return dists

    def loss(self, x, y):
        return self.loss_fn(self.forward(x), y)

    def update_dispersion(
        self,
        X_train: Union[np.ndarray, pd.DataFrame, torch.Tensor],
        y_train: Union[np.ndarray, pd.Series, torch.Tensor],
    ) -> None:
        X = self._to_tensor(X_train)
        y = self._to_tensor(y_train)
        disp = estimate_dispersion(self.distribution, self.forward(X), y, X.shape[1])
        self.dispersion = nn.Parameter(torch.Tensor([disp]), requires_grad=False)

    def mean(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the predicted means for the given observations, specific to the model type.
        """
        return self.forward(torch.Tensor(x)).detach().numpy().squeeze()
