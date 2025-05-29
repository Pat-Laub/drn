import abc
from typing import Any
import torch
import torch.nn as nn


class BaseModel(nn.Module, abc.ABC):
    learning_rate: float

    @abc.abstractmethod
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

    @abc.abstractmethod
    def distributions(self, x: torch.Tensor) -> Any: ...
