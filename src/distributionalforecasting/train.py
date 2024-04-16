import copy
from typing import Callable, List, Union

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange


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
    gradient_clipping=False,
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
                no_improvement = 0


            else:
                no_improvement += 1

            if print_details and epoch % log_interval == 0:
                tqdm.write(
                    f"Epoch {epoch} \t| batch train loss: {loss.item():.4f}"
                    + f"\t| validation loss:  {loss_val.item():.4f}"
                    + f"\t| no improvement: {no_improvement}"
                )

            if no_improvement > patience:
                if print_details:
                    tqdm.write("Stopping early!")
                break

    finally:
        # Load the best model found during training
        model.load_state_dict(best_model)
       
def split_and_preprocess(features, target, num_features, cat_features, seed = 42, num_standard = True):
    # Before preprocessing split
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
            features, target, random_state=seed, train_size=0.8, shuffle=True
    )

    x_train_raw, x_val_raw, y_train, y_val = train_test_split(
            x_train_raw, y_train, random_state=seed, train_size=0.75, shuffle=True
    )

    # Determine the full set of categories for each feature
    all_categories = {feature: set() for feature in cat_features}
    for feature in cat_features:
        all_categories[feature].update(features[feature].unique())

    # Convert each categorical feature to a categorical type with all possible categories
    for feature in cat_features:
        x_train_raw[feature] = pd.Categorical(x_train_raw[feature], categories=all_categories[feature])
        x_val_raw[feature] = pd.Categorical(x_val_raw[feature], categories=all_categories[feature])
        x_test_raw[feature] = pd.Categorical(x_test_raw[feature], categories=all_categories[feature])
        features[feature] = pd.Categorical(features[feature], categories=all_categories[feature])

    # One-hot Encoding
    features_one_hot = pd.get_dummies(features, columns=cat_features)
    features_one_hot = features_one_hot.astype(float)
        

    x_train, x_test, y_train, y_test = train_test_split(
            features_one_hot, target, random_state=seed, train_size=0.8, shuffle=True   
     )

    x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, random_state=seed, train_size=0.75, shuffle=True
    )

    if num_standard:
        # Standarise the numeric features.
        ct = ColumnTransformer(
                [("standardize", StandardScaler(), num_features)],
                remainder="passthrough",
                verbose_feature_names_out=False,
        )
            

        x_train = ct.fit_transform(x_train)
        x_val = ct.transform(x_val)
        x_test = ct.transform(x_test)
    else:
        ct = None
   
    x_train = pd.DataFrame(x_train, index = x_train_raw.index, columns = features_one_hot.columns)
    x_val = pd.DataFrame(x_val, index = x_val_raw.index, columns = features_one_hot.columns)
    x_test = pd.DataFrame(x_test, index = x_test_raw.index, columns = features_one_hot.columns)

    return(x_train, x_val, x_test, y_train, y_val, y_test,\
             x_train_raw, x_val_raw, x_test_raw,\
                  num_features, cat_features,
                      all_categories, ct)


