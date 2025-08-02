import abc
import tempfile
from typing import Any, Optional, Union
import warnings
import numpy as np
import pandas as pd
import torch
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from lightning.pytorch.utilities import disable_possible_user_warnings

disable_possible_user_warnings()


class BaseModel(L.LightningModule, abc.ABC):
    learning_rate: float

    @abc.abstractmethod
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def distributions(
        self, x: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor]
    ) -> Any: ...

    def training_step(self, batch: Any, idx: int) -> torch.Tensor:
        x, y = batch
        loss_batch = self.loss(x, y)
        self.log("train_loss", loss_batch, prog_bar=True)
        return loss_batch

    def validation_step(self, batch: Any, idx: int) -> torch.Tensor:
        x, y = batch
        loss_val = self.loss(x, y)
        self.log("val_loss", loss_val, prog_bar=True)
        return loss_val

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.DataFrame, pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
        batch_size: int = 128,
        epochs: int = 10,
        patience: int = 5,
        **trainer_kwargs,
    ) -> None:
        # Set some default trainer arguments
        trainer_kwargs.setdefault("max_epochs", epochs)
        trainer_kwargs.setdefault("accelerator", "cpu")
        trainer_kwargs.setdefault("devices", 1)
        trainer_kwargs.setdefault("logger", False)
        trainer_kwargs.setdefault("deterministic", True)
        trainer_kwargs.setdefault("enable_progress_bar", True)
        trainer_kwargs.setdefault("enable_model_summary", True)

        has_val = X_val is not None and y_val is not None

        # Build training DataLoader
        train_tensor = TensorDataset(
            self._to_tensor(X_train), self._to_tensor(y_train).squeeze()
        )
        train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

        # Simple train if no validation provided
        if not has_val:
            trainer_kwargs.setdefault("enable_checkpointing", False)
            trainer = L.Trainer(**trainer_kwargs)
            trainer.fit(self, train_loader)
            self.eval()
            return

        # Build validation DataLoader
        val_tensor = TensorDataset(
            self._to_tensor(X_val), self._to_tensor(y_val).squeeze()
        )
        val_loader = DataLoader(val_tensor, batch_size=len(val_tensor), shuffle=False)

        if trainer_kwargs.get("enable_checkpointing") is False:
            warnings.warn(
                "Early stopping requires checkpointing. "
                "Overriding enable_checkpointing=True.",
                UserWarning,
            )
        # force checkpointing on, ignore user override
        trainer_kwargs["enable_checkpointing"] = True

        # Validation: checkpointing + early stopping
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_cb = ModelCheckpoint(
                dirpath=tmpdir,
                filename="best",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            )
            early_cb = EarlyStopping(
                monitor="val_loss", mode="min", patience=patience, verbose=False
            )

            trainer = L.Trainer(callbacks=[ckpt_cb, early_cb], **trainer_kwargs)
            trainer.fit(self, train_loader, val_loader)

            # Restore best weights
            best_path = ckpt_cb.best_model_path
            if best_path:
                ckpt = torch.load(
                    best_path, map_location=lambda s, loc: s, weights_only=False
                )
                state = ckpt.get("state_dict", ckpt)
                self.load_state_dict(state)

        self.eval()

    def _to_tensor(
        self, arr: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor]
    ) -> torch.Tensor:
        """
        Convert input data to a PyTorch tensor.
        """
        if isinstance(arr, torch.Tensor):
            return arr
        elif isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series):
            arr = arr.values
        return torch.Tensor(arr, device=self.device)
