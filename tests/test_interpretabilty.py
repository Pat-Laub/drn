import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
from synthetic_dataset import generate_synthetic_data

from drn import *


def test_plot_adjustment_factors():

    X_train, Y_train, train_dataset, val_dataset = generate_synthetic_data()
    y_train = Y_train.cpu().numpy()

    x_train = pd.DataFrame(
        X_train.detach().cpu().numpy(),
        columns=[f"X_{i}" for i in range(X_train.shape[1])],
    )

    cutpoints = drn_cutpoints(0, 1, y_train, 0.1, 2)

    glm = GLM.from_statsmodels(X_train, Y_train, distribution="gamma")

    drn = DRN(X_train.shape[1], cutpoints, glm)
    train(
        drn,
        drn_loss,
        train_dataset,
        val_dataset,
        epochs=2,
    )

    drn_explainer = DRNExplainer(
        drn,
        glm,
        cutpoints,
        x_train,
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

    drn_explainer.cdf_plot(
        instance,
        method="Kernel",
        nsamples_background_fraction=0.5,
        top_K_features=3,
        labelling_gap=0.15,
        dist_property="90% Quantile",
        x_range=(2.5, 4.5),
        y_range=(0.87, 0.93),
        density_transparency=0.9,
        adjustment=True,
        shap_fontsize=40,
        plot_title="90% Quantile Adjustment Explanation",
        figsize=(20, 20),
    )
