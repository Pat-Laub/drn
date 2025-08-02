import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
from synthetic_dataset import generate_synthetic_data

from drn import (
    GLM,
    CANN,
    MDN,
    DDR,
    DRN,
    gamma_estimate_dispersion,
    merge_cutpoints,
    crps,
)


def _to_tensor(arr):
    """Convert NumPy array or pandas to torch.FloatTensor."""
    if isinstance(arr, (pd.DataFrame, pd.Series)):
        arr = arr.values
    return torch.tensor(arr, dtype=torch.float32)


def check_crps(model, X_train, y_train, grid_size=3000):
    X = _to_tensor(X_train)
    Y = _to_tensor(y_train)
    grid = torch.linspace(0, Y.max().item() * 1.1, grid_size).unsqueeze(-1).to(X.device)
    dists = model.distributions(X)
    cdfs = dists.cdf(grid)
    grid = grid.squeeze()
    crps_val = crps(Y, grid, cdfs)
    assert crps_val.shape == Y.shape
    assert crps_val.mean() > 0


def test_glm():
    print("\n\nTraining GLM\n")
    X_train, y_train, X_val, y_val = generate_synthetic_data()

    # --- 1) NUMPY inputs ---
    torch.manual_seed(1)
    glm_np = GLM(X_train.shape[1], distribution="gamma")
    glm_np.fit(X_train, y_train, X_val, y_val, epochs=2)
    glm_np.update_dispersion(_to_tensor(X_train), _to_tensor(y_train))
    check_crps(glm_np, X_train, y_train)

    # --- 2) PANDAS inputs (sanity check) ---
    Xtr_df = pd.DataFrame(X_train, columns=[f"X{i}" for i in range(X_train.shape[1])])
    ytr_sr = pd.Series(y_train, name="Y")
    Xvl_df = pd.DataFrame(X_val, columns=Xtr_df.columns)
    yvl_sr = pd.Series(y_val, name="Y")

    glm_pd = GLM(X_train.shape[1], distribution="gamma")
    # one short epoch just to confirm no errors
    glm_pd.fit(Xtr_df, ytr_sr, Xvl_df, yvl_sr, epochs=1)
    glm_pd.update_dispersion(_to_tensor(Xtr_df), _to_tensor(ytr_sr))
    check_crps(glm_pd, Xtr_df, ytr_sr)


def test_glm_from_statsmodels():
    X_train, y_train, _, _ = generate_synthetic_data()

    # tensor‐based
    glm1 = GLM.from_statsmodels(
        _to_tensor(X_train), _to_tensor(y_train), distribution="gamma"
    )
    d1 = gamma_estimate_dispersion(
        glm1.forward(_to_tensor(X_train)), _to_tensor(y_train), X_train.shape[1]
    )
    assert np.isclose(d1, glm1.dispersion.item())

    # numpy‐based
    glm2 = GLM.from_statsmodels(X_train, y_train, distribution="gamma").to(
        _to_tensor(X_train).device
    )
    d2 = gamma_estimate_dispersion(
        glm2.forward(_to_tensor(X_train)), _to_tensor(y_train), X_train.shape[1]
    )
    assert np.isclose(d2, glm2.dispersion.item())

    # pandas‐based, compare to statsmodels predictions
    X_df = pd.DataFrame(X_train, columns=[f"X{i}" for i in range(X_train.shape[1])])
    y_sr = pd.Series(y_train, name="Y")
    glm3 = GLM.from_statsmodels(X_df, y_sr, distribution="gamma").to(
        _to_tensor(X_df).device
    )

    sm_mod = sm.GLM(
        y_sr, sm.add_constant(X_df), family=Gamma(link=sm.families.links.Log())
    ).fit()
    sm_pred = sm_mod.predict(sm.add_constant(X_df))
    our_pred = glm3.forward(_to_tensor(X_df)).cpu().detach().numpy()
    assert np.allclose(sm_pred, our_pred)


def test_cann():
    X_train, y_train, X_val, y_val = generate_synthetic_data()

    torch.manual_seed(2)
    base = GLM(X_train.shape[1], distribution="gamma")
    base.fit(X_train, y_train, X_val, y_val, epochs=2)

    cann = CANN(base, num_hidden_layers=2, hidden_size=100)
    cann.fit(X_train, y_train, X_val, y_val, epochs=2)

    cann.update_dispersion(_to_tensor(X_train), _to_tensor(y_train))
    check_crps(cann, X_train, y_train)


def test_mdn():
    X_train, y_train, X_val, y_val = generate_synthetic_data()

    torch.manual_seed(3)
    mdn = MDN(X_train.shape[1], num_components=5, distribution="gamma")
    mdn.fit(X_train, y_train, X_val, y_val, epochs=2)

    check_crps(mdn, X_train, y_train)


def setup_cutpoints(y_train):
    Y = _to_tensor(y_train)
    max_y = Y.max().item()
    n = Y.numel()
    cps = np.linspace(0.0, max_y * 1.01, int(np.ceil(0.1 * n)))
    assert len(cps) >= 2
    return cps.tolist()


def test_ddr():
    X_train, y_train, X_val, y_val = generate_synthetic_data()
    cps = setup_cutpoints(y_train)

    torch.manual_seed(4)
    ddr = DDR(X_train.shape[1], cps, hidden_size=100)
    ddr.fit(X_train, y_train, X_val, y_val, epochs=2)

    check_crps(ddr, X_train, y_train)


def test_drn():
    X_train, y_train, X_val, y_val = generate_synthetic_data()
    cps0 = setup_cutpoints(y_train)
    cps1 = merge_cutpoints(cps0, y_train, min_obs=2)
    assert len(cps1) >= 2

    torch.manual_seed(5)
    base = GLM(X_train.shape[1], distribution="gamma")
    base.fit(X_train, y_train, X_val, y_val, epochs=2)
    base.update_dispersion(_to_tensor(X_train), _to_tensor(y_train))

    for loss_fn in ("jbce", "nll"):
        drn = DRN(
            base,
            cps1,
            hidden_size=100,
            loss_metric=loss_fn,
            kl_alpha=1.0,
            mean_alpha=1.0,
            dv_alpha=1.0,
            tv_alpha=1.0,
        )
        drn.fit(X_train, y_train, X_val, y_val, epochs=2)
        check_crps(drn, X_train, y_train)


def test_torch_api():
    X_train, y_train, X_val, y_val = generate_synthetic_data()
    cps = setup_cutpoints(y_train)

    torch.manual_seed(5)
    glm = GLM.from_statsmodels(
        _to_tensor(X_train), _to_tensor(y_train), distribution="gamma"
    )

    hs = 5
    drn1 = DRN(glm, cps, num_hidden_layers=2, hidden_size=hs)
    drn1.fit(X_train, y_train, X_val, y_val, epochs=2)
    total_w = sum(p.numel() for p in drn1.hidden_layers.parameters())
    expected = hs * X_train.shape[1] + hs + hs * hs + hs
    assert total_w == expected

    # numpy‐int hyperparameters
    drn2 = DRN(glm, cps, num_hidden_layers=np.int64(2), hidden_size=np.int64(hs))
    drn2.fit(X_train, y_train, X_val, y_val, epochs=2)
    total_w2 = sum(p.numel() for p in drn2.hidden_layers.parameters())
    assert total_w2 == expected and len(drn2.hidden_layers) // 3 == 2

    # dropout behaviour
    drn3 = DRN(glm, cps, num_hidden_layers=2, hidden_size=hs, dropout_rate=0.5)
    drn3.fit(X_train, y_train, X_val, y_val, epochs=2)

    drn3.train()
    p1 = drn3(_to_tensor(X_train))[-1]
    p2 = drn3(_to_tensor(X_train))[-1]
    assert not torch.allclose(p1, p2)

    drn3.eval()
    q1 = drn3(_to_tensor(X_train))[-1]
    q2 = drn3(_to_tensor(X_train))[-1]
    assert torch.allclose(q1, q2)
