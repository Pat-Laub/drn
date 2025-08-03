import os
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch
import lightning as L
from synthetic_dataset import generate_synthetic_tensordataset

from drn import *


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


def test_glm_checkpoint():
    X_train, Y_train, train_data, val_data = generate_synthetic_tensordataset()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)

    torch.manual_seed(1)
    glm_model = GLM(distribution="gamma", p=X_train.shape[1])
    trainer = L.Trainer(
        max_epochs=2, logger=False, enable_checkpointing=False, accelerator="cpu"
    )
    trainer.fit(glm_model, train_loader, val_loader)
    glm_model.update_dispersion(X_train, Y_train)

    # save to temp file
    with tempfile.NamedTemporaryFile(suffix=".ckpt") as tmp:
        trainer.save_checkpoint(tmp.name)
        glm_loaded = GLM.load_from_checkpoint(tmp.name)

    # compare weights
    for name, param in glm_model.named_parameters():
        lp = dict(glm_loaded.named_parameters())[name]
        assert torch.allclose(param, lp), f"GLM param {name} mismatch"

    # compare predictions via CDF
    glm_model.eval()
    glm_loaded.eval()
    d_orig = glm_model.distributions(X_train)
    d_loaded = glm_loaded.distributions(X_train)
    grid = Y_train.unsqueeze(-1)
    cdf_o = d_orig.cdf(grid).squeeze(-1)
    cdf_l = d_loaded.cdf(grid).squeeze(-1)
    assert torch.allclose(cdf_o, cdf_l), "GLM predictions mismatch"


def test_cann_checkpoint():
    X_train, Y_train, train_data, val_data = generate_synthetic_tensordataset()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)

    torch.manual_seed(2)
    glm = GLM(distribution="gamma", p=X_train.shape[1])
    trainer = L.Trainer(
        max_epochs=2, logger=False, enable_checkpointing=False, accelerator="cpu"
    )
    trainer.fit(glm, train_loader, val_loader)

    cann_model = CANN(glm, num_hidden_layers=2, hidden_size=100)
    trainer.fit(cann_model, train_loader, val_loader)
    cann_model.update_dispersion(X_train, Y_train)

    with tempfile.NamedTemporaryFile(suffix=".ckpt") as tmp:
        trainer.save_checkpoint(tmp.name)
        cann_loaded = CANN.load_from_checkpoint(tmp.name)

    # compare weights
    for name, param in cann_model.named_parameters():
        lp = dict(cann_loaded.named_parameters())[name]
        close = torch.allclose(param, lp)
        all_nan = torch.isnan(param).all() and torch.isnan(lp).all()
        assert close or all_nan, f"CANN param {name} mismatch"

    # compare predictions via CDF
    cann_model.eval()
    cann_loaded.eval()
    d_orig = cann_model.distributions(X_train)
    d_loaded = cann_loaded.distributions(X_train)
    grid = Y_train.unsqueeze(-1)
    cdf_o = d_orig.cdf(grid).squeeze(-1)
    cdf_l = d_loaded.cdf(grid).squeeze(-1)
    assert torch.allclose(cdf_o, cdf_l), "CANN predictions mismatch"


def test_mdn_checkpoint():
    X_train, Y_train, train_data, val_data = generate_synthetic_tensordataset()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)

    torch.manual_seed(3)
    mdn_model = MDN(num_components=5, distribution="gamma")
    trainer = L.Trainer(
        max_epochs=2, logger=False, enable_checkpointing=False, accelerator="cpu"
    )
    trainer.fit(mdn_model, train_loader, val_loader)

    with tempfile.NamedTemporaryFile(suffix=".ckpt") as tmp:
        trainer.save_checkpoint(tmp.name)
        mdn_loaded = MDN.load_from_checkpoint(tmp.name)

    # compare weights
    for name, param in mdn_model.named_parameters():
        lp = dict(mdn_loaded.named_parameters())[name]
        assert torch.allclose(param, lp), f"MDN param {name} mismatch"

    # compare predictions via CDF
    mdn_model.eval()
    mdn_loaded.eval()
    d_orig = mdn_model.distributions(X_train)
    d_loaded = mdn_loaded.distributions(X_train)
    grid = Y_train.unsqueeze(-1)
    cdf_o = d_orig.cdf(grid).squeeze(-1)
    cdf_l = d_loaded.cdf(grid).squeeze(-1)
    assert torch.allclose(cdf_o, cdf_l), "MDN predictions mismatch"


def test_ddr_checkpoint():
    X_train, Y_train, train_data, val_data = generate_synthetic_tensordataset()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)

    cutpoints = setup_cutpoints(Y_train)
    torch.manual_seed(4)
    ddr_model = DDR(cutpoints, hidden_size=100)
    trainer = L.Trainer(
        max_epochs=2, logger=False, enable_checkpointing=False, accelerator="cpu"
    )
    trainer.fit(ddr_model, train_loader, val_loader)

    with tempfile.NamedTemporaryFile(suffix=".ckpt") as tmp:
        trainer.save_checkpoint(tmp.name)
        ddr_loaded = DDR.load_from_checkpoint(tmp.name)

    # compare weights
    for name, param in ddr_model.named_parameters():
        lp = dict(ddr_loaded.named_parameters())[name]
        assert torch.allclose(param, lp), f"DDR param {name} mismatch"

    # compare predictions via CDF
    ddr_model.eval()
    ddr_loaded.eval()
    d_orig = ddr_model.distributions(X_train)
    d_loaded = ddr_loaded.distributions(X_train)
    grid = Y_train.unsqueeze(-1)
    cdf_o = d_orig.cdf(grid).squeeze(-1)
    cdf_l = d_loaded.cdf(grid).squeeze(-1)
    assert torch.allclose(cdf_o, cdf_l), "DDR predictions mismatch"


def test_drn_checkpoint():
    X_train, Y_train, train_data, val_data = generate_synthetic_tensordataset()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)

    cut_ddr = setup_cutpoints(Y_train)
    cut_drn = merge_cutpoints(cut_ddr, Y_train.cpu().numpy(), min_obs=2)

    torch.manual_seed(5)
    glm = GLM(distribution="gamma", p=X_train.shape[1])
    trainer = L.Trainer(
        max_epochs=2, logger=False, enable_checkpointing=False, accelerator="cpu"
    )
    trainer.fit(glm, train_loader, val_loader)
    glm.update_dispersion(X_train, Y_train)

    drn_model = DRN(glm, cut_drn, hidden_size=100)
    trainer.fit(drn_model, train_loader, val_loader)

    with tempfile.NamedTemporaryFile(suffix=".ckpt") as tmp:
        trainer.save_checkpoint(tmp.name)
        drn_loaded = DRN.load_from_checkpoint(tmp.name)

    # compare weights
    for name, param in drn_model.named_parameters():
        lp = dict(drn_loaded.named_parameters())[name]
        assert torch.allclose(param, lp), f"DRN param {name} mismatch"

    # compare predictions via CDF
    drn_model.eval()
    drn_loaded.eval()
    d_orig = drn_model.distributions(X_train)
    d_loaded = drn_loaded.distributions(X_train)
    grid = Y_train.unsqueeze(-1)
    cdf_o = d_orig.cdf(grid).squeeze(-1)
    cdf_l = d_loaded.cdf(grid).squeeze(-1)
    assert torch.allclose(cdf_o, cdf_l), "DRN predictions mismatch"


def test_torch_compat_checkpoint():
    X_train, Y_train, train_data, val_data = generate_synthetic_tensordataset()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)

    cutpoints = setup_cutpoints(Y_train)
    torch.manual_seed(5)
    glm = GLM.from_statsmodels(X_train, Y_train, distribution="gamma")

    drn_model = DRN(glm, cutpoints, num_hidden_layers=2, hidden_size=5)
    trainer = L.Trainer(
        max_epochs=2, logger=False, enable_checkpointing=False, accelerator="cpu"
    )
    trainer.fit(drn_model, train_loader, val_loader)

    with tempfile.NamedTemporaryFile(suffix=".ckpt") as tmp:
        trainer.save_checkpoint(tmp.name)
        drn_loaded = DRN.load_from_checkpoint(tmp.name)

    # compare weights
    for name, param in drn_model.named_parameters():
        lp = dict(drn_loaded.named_parameters())[name]
        assert torch.allclose(param, lp), f"Torch-compat DRN param {name} mismatch"

    # compare predictions via CDF
    drn_model.eval()
    drn_loaded.eval()
    d_orig = drn_model.distributions(X_train)
    d_loaded = drn_loaded.distributions(X_train)
    grid = Y_train.unsqueeze(-1)
    cdf_o = d_orig.cdf(grid).squeeze(-1)
    cdf_l = d_loaded.cdf(grid).squeeze(-1)
    assert torch.allclose(cdf_o, cdf_l), "Torch-compat DRN predictions mismatch"
