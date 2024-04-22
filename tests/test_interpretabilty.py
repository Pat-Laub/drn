import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
from synthetic_dataset import generate_synthetic_data

import distributionalforecasting as df


def test_plot_adjustment_factors():

    X_train, Y_train, train_dataset, val_dataset = generate_synthetic_data()
    y_train = Y_train.cpu().numpy()

    cutpoints = df.drn_cutpoints(0, 1, 0.1, y_train, 2)

    glm = df.GLM.from_statsmodels(X_train, Y_train, distribution="gamma")

    drn = df.DRN(X_train.shape[1], cutpoints, glm)
    df.train(
        drn,
        df.drn_jbce_loss,
        train_dataset,
        val_dataset,
        epochs=2,
    )

    drn_explainer = df.DRNExplainer(
        drn,
        glm,
        cutpoints,
        X_train,
        cat_features=[],
    )

    instance = pd.DataFrame(
        np.array([[0.0, 1.0, 2.0, 3.0]]), columns=["X_1", "X_2", "X_3", "X_4"]
    )

    drn_explainer.plot_adjustment_factors(
        instance,
        num_interpolations=3000,
        plot_adjustments_labels=False,
        plot_mean_adjustment=True,
    )

    instance = np.array([[0.0, 1.0, 2.0, 3.0]])

    drn_explainer.plot_adjustment_factors(
        instance,
        num_interpolations=3000,
        plot_adjustments_labels=False,
        plot_mean_adjustment=True,
    )
