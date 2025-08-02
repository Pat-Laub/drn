import os


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch
from synthetic_dataset import generate_synthetic_tensordataset
from synthetic_dataset import generate_synthetic_data
from drn import GLM, CANN, MDN, train

# force CPU in .fit
FIT_KW = dict(
    accelerator="cpu", devices=1, deterministic=True, enable_checkpointing=False
)


def _extract_numpy_from_tensordataset(ds):
    # ds is a torch.utils.data.TensorDataset
    X_t, y_t = ds.tensors
    return X_t.cpu().numpy(), y_t.cpu().numpy()


def _compare_params(m1, m2, atol=1e-6):
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        close = torch.allclose(p1, p2, atol=atol)
        both_nan = torch.isnan(p1).all() and torch.isnan(p2).all()
        assert (
            close or both_nan
        ), f"Parameters are not close enough: {p1} vs {p2} (close={close}, both_nan={both_nan})"


def test_glm_train_vs_fit_equivalence():
    """GLM trained via `train(...)` vs. via `.fit(...)` should end up identical."""
    seed = 42
    # get torch tensors + datasets
    X_t, y_t, train_ds, val_ds = generate_synthetic_tensordataset()
    # extract numpy arrays for .fit
    X_train_np, y_train_np = _extract_numpy_from_tensordataset(train_ds)
    X_val_np, y_val_np = _extract_numpy_from_tensordataset(val_ds)

    # 1) train(...) version
    torch.manual_seed(seed)
    glm1 = GLM(X_t.shape[1], distribution="gamma")
    train(glm1, train_ds, val_ds, epochs=1)

    # 2) .fit(...) version
    torch.manual_seed(seed)
    glm2 = GLM(X_t.shape[1], distribution="gamma")
    glm2.fit(X_train_np, y_train_np, X_val_np, y_val_np, epochs=1, **FIT_KW)

    # compare every learned parameter
    _compare_params(glm1, glm2)


def test_cann_train_vs_fit_equivalence():
    """CANN trained via `train(...)` vs. via `.fit(...)` should end up identical."""
    seed = 1234
    # get torch tensors + datasets
    X_t, y_t, train_ds, val_ds = generate_synthetic_tensordataset()
    # extract numpy arrays for .fit
    X_train_np, y_train_np = X_t.cpu().numpy(), y_t.cpu().numpy()
    X_val_np, y_val_np = _extract_numpy_from_tensordataset(val_ds)

    # 1) train(...) version
    torch.manual_seed(seed)
    glm1 = GLM(X_t.shape[1], distribution="gamma")
    cann1 = CANN(glm1, num_hidden_layers=1, hidden_size=16)
    train(cann1, train_ds, val_ds, epochs=1)

    # 2) .fit(...) version
    torch.manual_seed(seed)
    glm2 = GLM(X_t.shape[1], distribution="gamma")
    cann2 = CANN(glm2, num_hidden_layers=1, hidden_size=16)
    cann2.fit(X_train_np, y_train_np, X_val_np, y_val_np, epochs=1, **FIT_KW)

    # compare every learned parameter
    _compare_params(cann1, cann2)


def test_mdn_train_vs_fit_equivalence():
    """MDN trained via `train(...)` vs. via `.fit(...)` should end up identical."""
    seed = 2025
    X_t, y_t, train_ds, val_ds = generate_synthetic_tensordataset()
    X_train_np, y_train_np = X_t.cpu().numpy(), y_t.cpu().numpy()
    X_val_np, y_val_np = _extract_numpy_from_tensordataset(val_ds)

    # 1) train(...) version
    torch.manual_seed(seed)
    glm1 = GLM(X_t.shape[1], distribution="gamma")
    mdn1 = MDN(X_t.shape[1], num_components=4, distribution="gamma")
    # wrap MDN into train helper exactly like before
    train(mdn1, train_ds, val_ds, epochs=1)

    # 2) .fit(...) version
    torch.manual_seed(seed)
    glm2 = GLM(X_t.shape[1], distribution="gamma")
    mdn2 = MDN(X_t.shape[1], num_components=4, distribution="gamma")
    mdn2.fit(X_train_np, y_train_np, X_val_np, y_val_np, epochs=1, **FIT_KW)

    _compare_params(mdn1, mdn2)


def test_glm_train_vs_fit_early_stopping_equivalence():
    """GLM: train(...) vs .fit(...) with early stopping should still match."""
    seed = 42
    # get torch tensors + datasets
    X_t, y_t, train_ds, val_ds = generate_synthetic_tensordataset()
    X_train_np, y_train_np = _extract_numpy_from_tensordataset(train_ds)
    X_val_np, y_val_np = _extract_numpy_from_tensordataset(val_ds)

    # 1) train(...) version with patience=1
    torch.manual_seed(seed)
    glm1 = GLM(X_t.shape[1], distribution="gamma")
    train(glm1, train_ds, val_ds, epochs=10, patience=3)

    # 2) .fit(...) version with the same epochs & patience
    torch.manual_seed(seed)
    glm2 = GLM(X_t.shape[1], distribution="gamma")
    glm2.fit(
        X_train_np, y_train_np, X_val_np, y_val_np, epochs=10, patience=3, **FIT_KW
    )

    # compare every learned parameter
    _compare_params(glm1, glm2)
