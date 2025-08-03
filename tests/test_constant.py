import pytest
import torch
import numpy as np
import pandas as pd

from drn import Constant, InverseGaussian, GLM
from torch.distributions import Gamma, Normal


def test_initial_state_values():
    # Gamma and IG should start with mean=1, dispersion=1; Gaussian with mean=0
    c_gamma = Constant("gamma")
    assert c_gamma.mean_value.item() == pytest.approx(1.0)
    assert c_gamma.dispersion.item() == pytest.approx(1.0)

    c_ig = Constant("inversegaussian")
    assert c_ig.mean_value.item() == pytest.approx(1.0)
    assert c_ig.dispersion.item() == pytest.approx(1.0)

    c_gauss = Constant("gaussian")
    assert c_gauss.mean_value.item() == pytest.approx(0.0)
    assert c_gauss.dispersion.item() == pytest.approx(1.0)


def test_distributions_initial():
    # Use a dummy input of batch size 4
    x = torch.zeros((4, 2))
    # Gamma
    c = Constant("gamma")
    dist = c.distributions(x)
    assert isinstance(dist, Gamma)
    # mean = alpha/beta = (1/phi)/(mean_value) = 1/1 = 1
    assert torch.allclose(dist.mean, torch.tensor([1.0] * 4))

    # IG
    c = Constant("inversegaussian")
    dist = c.distributions(x)
    assert isinstance(dist, InverseGaussian)
    # mean property for custom IG
    assert torch.allclose(torch.tensor(dist.mean), torch.tensor([1.0] * 4))

    # Gaussian
    c = Constant("gaussian")
    dist = c.distributions(x)
    assert isinstance(dist, Normal)
    assert torch.allclose(dist.mean, torch.tensor([0.0] * 4))


def test_fit_updates_mean_and_dispersion_gamma():
    # Create sample y with non-equal values for Gamma
    y = np.array([2.0, 4.0, 6.0, 8.0], dtype=float)
    c = Constant("gamma")
    # fit ignores X
    c.fit(X_train=None, y_train=y)
    # mean should update
    assert c.mean_value.item() == pytest.approx(5.0)
    # dispersion = sum((y-5)^2/5^2)/(n-1)
    expected_phi = np.sum((y - 5.0) ** 2 / 5.0**2) / (len(y) - 1)
    assert c.dispersion.item() == pytest.approx(expected_phi)


def test_fit_updates_mean_and_dispersion_gaussian():
    # Gaussian dispersion is sigma (std dev)
    y = np.array([1.0, 3.0, 5.0], dtype=float)
    c = Constant("gaussian")
    c.fit(X_train=None, y_train=y)
    # mean = 3.0
    assert c.mean_value.item() == pytest.approx(3.0)
    # sigma = sqrt(sum((y-3)^2)/(n-1)) = sqrt((4+0+4)/2) = sqrt(4) = 2
    assert c.dispersion.item() == pytest.approx(2.0)


def test_loss_decreases_after_fit():
    # For a simple dataset, loss before fit is higher than after fit
    # Note: we need y to have some variance for dispersion estimation to be > 0
    y = np.array([2.0, 3.0, 4.0, 5.0], dtype=float)
    x = torch.zeros((4, 1))
    c = Constant("gamma")
    # initial loss with mean=1
    loss_before = c.loss(x, torch.tensor(y))
    c.fit(X_train=None, y_train=y)
    loss_after = c.loss(x, torch.tensor(y))
    assert loss_after < loss_before


def test_invalid_distribution_raises():
    with pytest.raises(ValueError):
        Constant("unsupported")


def test_equivalence_with_default_glm():
    # Constant without fit should match GLM without fit
    for dist in ("gamma", "gaussian", "inversegaussian"):
        const = Constant(dist)
        glm = GLM(p=2, distribution=dist)
        # Dummy input batch of size 3
        x = torch.randn((3, 2))
        # forward matches
        out_c = const.forward(x)
        out_g = glm.forward(x)
        assert torch.allclose(out_c, out_g)
        # dispersion matches
        assert const.dispersion.item() == pytest.approx(glm.dispersion.item())
        # distribution parameters match
        d1 = const.distributions(x)
        d2 = glm.distributions(x)
        if dist == "gamma":
            assert torch.allclose(d1.concentration, d2.concentration)
            assert torch.allclose(d1.rate, d2.rate)
        elif dist == "gaussian":
            assert torch.allclose(d1.loc, d2.loc)
            assert torch.allclose(d1.scale, d2.scale)
        else:
            assert torch.allclose(torch.tensor(d1.mean), torch.tensor(d2.mean))
            assert const.dispersion.item() == pytest.approx(glm.dispersion.item())


def test_equivalence_with_null_glm_constructor():
    # GLM null_model=True constructor also sets zero coeffs & unit dispersion
    for dist in ("gamma", "gaussian", "inversegaussian"):
        const = Constant(dist)
        glm = GLM(p=2, distribution=dist)
        x = torch.randn((4, 2))
        assert torch.allclose(const.forward(x), glm.forward(x))
        assert const.dispersion.item() == pytest.approx(glm.dispersion.item())


def test_equivalence_with_from_statsmodels_null_model():
    # Compare Constant against GLM.from_statsmodels null_model=True
    for dist in ("gamma", "gaussian", "inversegaussian"):
        # sample data
        n, p = 6, 3
        y = np.linspace(1.0, 3.0, n, dtype=float)
        x = torch.randn((n, p))

        # fit constant
        const = Constant(dist)
        const.fit(X_train=None, y_train=y)

        # obtain statsmodels null GLM
        glm_null = GLM.from_statsmodels(
            x, torch.from_numpy(y).float(), distribution=dist, null_model=True
        )
        glm_null.eval()

        # predictions should match
        pred_c = const.forward(x)
        pred_g = glm_null.forward(x)
        assert torch.allclose(pred_c, pred_g, atol=1e-6)

        # distribution means match
        d_c = const.distributions(x)
        d_g = glm_null.distributions(x)
        mean_c = d_c.mean if hasattr(d_c, "mean") else torch.tensor(d_c.mean)
        mean_g = d_g.mean if hasattr(d_g, "mean") else torch.tensor(d_g.mean)
        assert torch.allclose(mean_c, mean_g, atol=1e-6)
