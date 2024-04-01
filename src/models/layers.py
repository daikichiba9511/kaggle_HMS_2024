import math
from collections.abc import Callable
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

logger = getLogger(__name__)


def gem(x: torch.Tensor, p: int = 3, eps: float = 1e-6) -> torch.Tensor:
    return F.avg_pool2d(x.clamp(min=eps).pow(p), kernel_size=(x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, eps={self.eps})"


class SEScale(nn.Module):
    def __init__(self, in_channels: int, r: int) -> None:
        super().__init__()
        if in_channels < r:
            logger.warning("r should be larger than in_channels")
        self.fc1 = nn.Linear(in_channels, r)
        self.fc2 = nn.Linear(r, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h)
        h = h.sigmoid()
        return x * h


class MixupForwarder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, lam: float, y: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = lam * x + (1 - lam) * x.flip(0)
        out = {"x": x}
        if y is not None:
            y = lam * y + (1 - lam) * y.flip(0)
            out["y"] = y
        return out


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


class MyAmplitudeToDB(torch.nn.Module):
    def __init__(
        self,
        stype: str = "power",
        top_db: float | None = None,
        amin: float = 1e-10,
        ref_value: float | Callable = 1.0,
    ) -> None:
        super().__init__()
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError("top_db must be positive value")
        self.top_db = top_db
        self.multiplier = 10.0 if stype == "power" else 20.0
        self.amin = amin
        self.ref_value = ref_value

        if isinstance(ref_value, float):
            self.db_multiplier = math.log10(max(self.amin, ref_value))
        else:
            self.db_multiplier = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.db_multiplier is None and callable(self.ref_value):
            self.db_multiplier = math.log10(max(self.amin, self.ref_value(x)))

        if self.db_multiplier is None:
            raise ValueError("ref_value must be float or callable")

        return torchaudio.functional.amplitude_to_DB(x, self.multiplier, self.amin, self.db_multiplier, self.top_db)
