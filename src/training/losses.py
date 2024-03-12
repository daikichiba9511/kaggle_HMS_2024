from typing import Callable, Literal

import torch

LossType = Literal["BCEWithLogitsLoss", "CrossEntropyLoss", "KLDivLoss"]
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def init_loss_fn(loss_name: str) -> LossFn:
    if loss_name == "BCEWithLogitsLoss":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        return loss_fn
    elif loss_name == "CrossEntropyLoss":
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn
    elif loss_name == "KLDivLoss":
        loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        return loss_fn
    else:
        raise NotImplementedError
