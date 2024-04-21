import os
import torch
from synthetic_dataset import generate_synthetic_data

import distributionalforecasting as df


def test_glm_mean():
    X_train, Y_train, _, _ = generate_synthetic_data()

    torch.manual_seed(1)
    glm = df.GLM(X_train.shape[1], distribution="gamma")

    # Either we call 'train' to move to same device as 'X_train', or do it manually
    glm = glm.to(X_train.device)

    glm.update_dispersion(X_train, Y_train)

    mean1 = glm.mean(X_train)
    mean2 = glm.distributions(X_train).mean
    assert torch.allclose(mean1, mean2)


def test_glm_save_load():
    X_train, Y_train, _, _ = generate_synthetic_data()

    torch.manual_seed(1)
    glm = df.GLM.from_statsmodels(X_train, Y_train, distribution="gamma")

    torch.save(glm.state_dict(), "glm.pt")

    glm_load = df.GLM(X_train.shape[1], distribution="gamma").to(X_train.device)
    glm_load.load_state_dict(torch.load("glm.pt"))

    os.remove("glm.pt")

    assert glm.state_dict().keys() == glm_load.state_dict().keys()
    for key in glm.state_dict().keys():
        assert torch.allclose(glm.state_dict()[key], glm_load.state_dict()[key])

    assert glm.dispersion == glm_load.dispersion
