import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


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


def init_spec_module(
    hop_length: int = 10_000 // 256,
    n_fft: int = 1024,
    win_length: int = 128,
    power: float | None = 2,
) -> torchaudio.transforms.Spectrogram:
    module = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="constant",
        power=power,
    )
    return module


def init_db_transform_module(stype: str = "power", top_db: int = 80) -> torchaudio.transforms.AmplitudeToDB:
    module = torchaudio.transforms.AmplitudeToDB(stype=stype, top_db=top_db)
    return module
