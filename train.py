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

import copy

# Define Criterion type hint
Criterion = Union[
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    Callable[[List[torch.Tensor], torch.Tensor], torch.Tensor]
]

def train(
    model: nn.Module,
    criterion: Criterion,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    epochs=1000,
    patience=50,
    lr=0.0005,
    device=None,
    log_interval=50,
    batch_size=None,
    optimizer=torch.optim.Adam,
    print_details=True,
    gradient_clipping=True,
) -> None:
    """
    A generic training function given a model and a criterion & optimizer.
    """
    if batch_size is None:
        batch_size = len(train_dataset)  # type: ignore
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))  # type: ignore

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        # Temporarily disable MPS which doesn't support some operations we need.
        # elif torch.backends.mps.is_available():
        #     device = torch.device("mps")
        else:
            device = torch.device("cpu")

    try:
        if print_details:
            print(f"Using device: {device}")
        model.to(device)

        optimizer = optimizer(model.parameters(), lr=lr)

        no_improvement = 0
        best_loss = torch.Tensor([float("inf")]).to(device)
        best_model = model.state_dict()

        range_selection = trange if print_details else range
        for epoch in range_selection(1, epochs + 1):
            model.train()

            for data, target in train_loader:
                x = data.to(device)
                y = target.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()

                if gradient_clipping:
                    # Check for NaN gradients
                    for name, param in model.named_parameters():
                        if param.grad is not None:  # Parameters might not have gradients if they are not trainable
                            if torch.isnan(param.grad).any():
                                raise ValueError(f"Gradient NaN detected in {name}. Try smaller learning rates.")

                    # If no error is raised, proceed with gradient clipping and optimizer step
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


                optimizer.step()

            model.eval()

            loss_val = torch.zeros(1, device=device)

            for data, target in val_loader:
                x = data.to(device)
                y = target.to(device)

                loss_val += criterion(model(x), y)
           
            if loss_val < best_loss:
                best_loss = loss_val
                best_model = copy.deepcopy(model.state_dict())
                #best_model = model.state_dict()
                no_improvement = 0


            else:
                no_improvement += 1

            if epoch % log_interval == 0:
                tqdm.write(
                    f"Epoch {epoch} \t| batch train loss: {loss.item():.4f}"
                    + f"\t| validation loss:  {loss_val.item():.4f}"
                    + f"\t| no improvement: {no_improvement}"
                )

            if no_improvement > patience:
                if print_details:
                    tqdm.write("Stopping early!")
                break  # Exit the loop

    finally:
        # Load the best model found during training
        model.load_state_dict(best_model)
       

