import torch
from synthetic_dataset import generate_synthetic_data

import distributionalforecasting as df


def test_glm():
    print("\n\nTraining GLM\n")
    X_train, Y_train, _, _ = generate_synthetic_data()

    torch.manual_seed(1)
    glm = df.GLM(X_train.shape[1], distribution='gamma')

    # Either we call 'train' to move to same device as 'X_train', or do it manually
    glm = glm.to(X_train.device)
    
    glm.dispersion = df.gamma_estimate_dispersion(
        glm.forward(X_train), Y_train, X_train.shape[1]
    )

    mean1 = glm.mean(X_train)
    mean2 = glm.distributions(X_train).mean
    assert torch.allclose(mean1, mean2)

