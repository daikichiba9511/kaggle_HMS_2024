import torch
import torch.nn as nn
import torch.nn.functional as F


def gem(x: torch.Tensor, p: int = 3, eps: float = 1e-6) -> torch.Tensor:
    return F.avg_pool2d(x.clamp(min=eps).pow(p), kernel_size=(x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, eps={self.eps})"
