import abc
from typing import Any
import torch
import lightning as L


class BaseModel(L.LightningModule, abc.ABC):
    learning_rate: float

    @abc.abstractmethod
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def distributions(self, x: torch.Tensor) -> Any: ...

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
