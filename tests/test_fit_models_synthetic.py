import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch
from synthetic_dataset import generate_synthetic_data

import distributionalforecasting as df


def test_glm():
    print("\n\nTraining GLM\n")
    X_train, Y_train, train_dataset, val_dataset = generate_synthetic_data()

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
    X_train, Y_train, train_dataset, val_dataset = generate_synthetic_data()

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
    X_train, Y_train, train_dataset, val_dataset = generate_synthetic_data()

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
    X_train, Y_train, train_dataset, val_dataset = generate_synthetic_data()

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
    X_train, Y_train, train_dataset, val_dataset = generate_synthetic_data()
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