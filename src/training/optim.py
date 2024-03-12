import logging
from typing import Literal, TypeAlias, TypedDict

import torch
from timm import scheduler as timm_scheduler

logger = logging.getLogger(__name__)


OptimizerType = Literal["AdamW"]


class AdamWParams(TypedDict):
    lr: float
    weight_decay: float
    eps: float


OptimParams: TypeAlias = AdamWParams


def init_optimizer(optimizer_name: str, model: torch.nn.Module, params: OptimParams) -> torch.optim.Optimizer:
    """optimizerの初期化
    Args:
        optimizer_name: optimizerの名前
        model: T<:nn.Module
        params: optimizerのパラメータ

    Returns:
        optimizer
    """
    logger.info(f"Initialize optimizer: {optimizer_name} (params: {params})")

    if optimizer_name == "AdamW":
        model_params = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        model_params = [
            {
                "params": [p for n, p in model_params if not any(nd in n for nd in no_decay)],
                "weight_decay": params["weight_decay"],
            },
            {
                "params": [p for n, p in model_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(params=model_params, lr=params["lr"], fused=False, eps=params["eps"])
        return optimizer
    else:
        raise NotImplementedError


SchdulerType = Literal["CosineLRScheduler"]


class CosineLRSchedulerParams(TypedDict):
    t_initial: int
    """何epochで1サイクルを終えるか"""
    lr_min: float
    """最小の学習率. よく使う値: 1e-6, 1e-7"""
    warmup_prefix: bool
    """warmupの前に1サイクルを終えるか."""
    warmup_t: int
    """warmupのepoch数. よく使う値: 0, 1, 2, 3"""
    warmup_lr_init: float
    """warmupの初期学習率. よく使う値: 1e-6, 1e-7"""
    cycle_limit: int
    """何サイクルまで行うか."""


SchduleParams: TypeAlias = CosineLRSchedulerParams


def init_scheduler(
    scheduler_name: str, optimizer: torch.optim.Optimizer, params: SchduleParams
) -> torch.optim.lr_scheduler.LRScheduler | timm_scheduler.cosine_lr.Scheduler:
    """schedulerの初期化

    Args:
        secheduler_name: schedulerの名前
        optimizer: T<:torch.optim.Optimizer
        params: schedulerのパラメータ

    """
    logger.info(f"Initialize optimizer: {scheduler_name} (params: {params})")

    if scheduler_name == "CosineLRScheduler":
        scheduler = timm_scheduler.CosineLRScheduler(
            optimizer,
            t_initial=params["t_initial"],
            lr_min=params["lr_min"],
            warmup_prefix=params["warmup_prefix"],
            warmup_lr_init=params["warmup_lr_init"],  # type: ignore
            warmup_t=params["warmup_t"],
            cycle_limit=params["cycle_limit"],
        )
        return scheduler
    else:
        raise NotImplementedError
