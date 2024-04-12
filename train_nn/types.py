from typing import Callable, Union, List
import torch

# Define Criterion type hint
Criterion = Union[
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    Callable[[List[torch.Tensor], torch.Tensor], torch.Tensor]
]
