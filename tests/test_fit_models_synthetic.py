import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import distributionalforecasting as df

import numpy as np
import pandas as pd
import torch

def generate_synthetic_data(n=1000, seed=1) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    x = rng.random(size=(n, 4))
    epsilon = rng.normal(0, 0.2, n)

    means = np.exp(
        0
        - 0.5 * x[:, 0]
        + 0.5 * x[:, 1]
        + np.sin(np.pi * x[:, 0])
        - np.sin(np.pi * np.log(x[:, 2] + 1))
        + np.cos(x[:, 1] * x[:, 2])
    ) + np.cos(x[:, 1])
    dispersion = 0.5

    y_lognormal = np.exp(rng.normal(means / 4, dispersion))
    y_gamma = rng.gamma(1 / dispersion, scale=dispersion * means / 4)

    Y = y_gamma * 0.5 + y_lognormal * 0.5 + (epsilon) ** 2
    return pd.DataFrame(x, columns=["X_1", "X_2", "X_3", "X_4"]), pd.Series(Y, name="Y")

def setup():
    dataset_size = 100

    x_all, y_all = generate_synthetic_data(dataset_size)

    num_val = int(dataset_size * 0.2)
    num_test = int(dataset_size * 0.2)
    num_train = dataset_size - num_val - num_test

    x_train, y_train = x_all[:num_train], y_all[:num_train]
    x_val, y_val = (
        x_all[num_train : num_train + num_val],
        y_all[num_train : num_train + num_val],
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    # Temporarily disable MPS which doesn't support some operations we need.
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    X_train = torch.Tensor(x_train.values).to(device)
    Y_train = torch.Tensor(y_train.values).to(device)
    X_val = torch.Tensor(x_val.values).to(device)
    Y_val = torch.Tensor(y_val.values).to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)

    return X_train, Y_train, train_dataset, val_dataset

def test_glm():
    print("\n\nTraining GLM\n")
    X_train, Y_train, train_dataset, val_dataset = setup()

    torch.manual_seed(1)
    glm = df.GLM(X_train.shape[1], distribution='gamma')

    df.train(
        glm,
        df.gamma_deviance_loss,
        train_dataset,
        val_dataset,
        epochs=2,
    )

    glm.dispersion = df.gamma_estimate_dispersion(
        glm.forward(X_train), Y_train, X_train.shape[1]
    )

def test_cann():
    print("\n\nTraining CANN\n")
    X_train, Y_train, train_dataset, val_dataset = setup()

    torch.manual_seed(2)
    glm = df.GLM(X_train.shape[1], distribution='gamma')
    df.train(
        glm,
        df.gamma_deviance_loss,
        train_dataset,
        val_dataset,
        epochs=2,
    )

    cann = df.CANN(glm, num_hidden_layers=2, hidden_size=100)
    df.train(
        cann,
        df.gamma_deviance_loss,
        train_dataset,
        val_dataset,
        epochs=2,
    )

    cann.dispersion = df.gamma_estimate_dispersion(cann.forward(X_train), Y_train, cann.p)

def test_mdn():
    print("\n\nTraining MDN\n")
    X_train, Y_train, train_dataset, val_dataset = setup()

    torch.manual_seed(3)
    mdn = df.MDN(X_train.shape[1], num_components=5, distribution='gamma')
    df.train(
        mdn,
        df.gamma_mdn_loss,
        train_dataset,
        val_dataset,
        epochs=2,
    )

def setup_cutpoints(Y_train):
    max_y = torch.max(Y_train).item()
    len_y = torch.numel(Y_train)
    c_0 = 0.0
    c_K = max_y * 1.01
    p = 0.1
    num_cutpoints = int(np.ceil(p * len_y))
    cutpoints_ddr = list(np.linspace(c_0, c_K, num_cutpoints))
    assert len(cutpoints_ddr) >= 2
    return cutpoints_ddr

def test_ddr():
    print("\n\nTraining DDR\n")
    X_train, Y_train, train_dataset, val_dataset = setup()

    cutpoints_ddr = setup_cutpoints(Y_train)

    torch.manual_seed(4)
    ddr = df.DDR(X_train.shape[1], cutpoints_ddr, hidden_size=100)
    df.train(
        ddr,
        df.ddr_loss,
        train_dataset,
        val_dataset,
        epochs=2
    )

def test_drn():
    print("\n\nTraining DRN\n")
    X_train, Y_train, train_dataset, val_dataset = setup()
    y_train = Y_train.cpu().numpy()
    
    cutpoints_ddr = setup_cutpoints(Y_train)
    cutpoints_drn = df.merge_cutpoints(cutpoints_ddr, y_train, min_obs=2)
    assert len(cutpoints_drn) >= 2

    torch.manual_seed(5)
    glm = df.GLM(X_train.shape[1], distribution='gamma')
    df.train(
        glm,
        df.gamma_deviance_loss,
        train_dataset,
        val_dataset,
        epochs=2,
    )
    glm.dispersion = df.gamma_estimate_dispersion(
        glm.forward(X_train), Y_train, X_train.shape[1]
    )

    drn = df.DRN(X_train.shape[1], cutpoints_drn, glm, hidden_size=100)
    df.train(
        drn,
        df.drn_jbce_loss,
        train_dataset,
        val_dataset,
        epochs=2,
    )