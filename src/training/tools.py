import functools
import logging
import pathlib
import time
from typing import Any, Callable, Literal, Protocol, Sequence, TypeAlias, TypedDict, TypeVar, cast

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from matplotlib import axes, figure
from matplotlib import pyplot as plt
from timm import scheduler as timm_scheduler
from timm import utils as timm_utils
from torch import optim as torch_optim
from torch.nn import utils as nn_utils
from torch.utils import data as torch_data

import wandb
from src.models import common as my_model_common
from src.training import losses as my_losses
from src.training import optim as my_optim
from src.utils import common as my_common_utils

logger = logging.getLogger(__name__)


class ClipParams(TypedDict):
    type: Literal["value", "norm"]
    value: float


def backward_step(
    batch_idx: int,
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: amp.grad_scaler.GradScaler | None,
    model: nn.Module,
    clip_params: ClipParams | None = None,
    grad_accum_steps: int = 1,
) -> None:
    if clip_params is None:
        clip_params = {"type": "norm", "value": 1000.0}

    if not torch.isfinite(loss).all() or torch.isnan(loss).any():
        logger.info("Loss includes infinite / nan. Skip updating !! ")
        return None

    if scaler is not None:
        scaled_loss = scaler.scale(loss)
        assert isinstance(scaled_loss, torch.Tensor)
        scaled_loss.backward()

        if clip_params["type"] == "value":
            nn_utils.clip_grad.clip_grad_value_(model.parameters(), clip_params["value"])
        elif clip_params["type"] == "norm":
            nn_utils.clip_grad.clip_grad_norm_(model.parameters(), clip_params["value"])
        else:
            raise ValueError(f"{clip_params = }")

        if batch_idx % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
    else:
        if batch_idx % grad_accum_steps == 0:
            loss.backward()
            optimizer.step()

    optimizer.zero_grad()


class AverageMeter:
    """Computes and stores the average and current value"""

    val: float | int
    avg: float | int
    sum: float | int
    count: int
    rows: list[float | int]

    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    def __str__(self) -> str:
        return f"Metrics {self.name}: Avg {self.avg}"

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.raws: list[float | int] = []

    def update(self, value: float | int, n: int = 1) -> None:
        if value in [np.nan, np.inf, -np.inf, float("inf"), float("-inf"), float("nan")]:
            logger.info("Skip nan or inf value")
            return None
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        self.raws.append(value)

    def to_dict(self) -> dict[str, list[float | int] | str | float]:
        return {
            "name": self.name,
            "avg": self.avg,
            "raw_values": self.raws,
        }


EarlyStopDirection = Literal["maximize", "minimize"]


class EarlyStopping:
    def __init__(
        self,
        patience: int,
        direction: EarlyStopDirection,
        delta: float = 0.0,
    ) -> None:
        if direction not in ["maximize", "minimize"]:
            raise ValueError(f"{direction = }")

        self._patience = patience
        self._direction = direction
        self._counter = 0
        self._best_score = float("-inf") if direction == "maximize" else float("inf")
        self._delta = delta
        self.is_improved = False

    def _is_improved(self, score: float, best_score: float) -> bool:
        if self._direction == "maximize":
            self.is_improved = score + self._delta > best_score
            return self.is_improved
        else:
            self.is_improved = score + self._delta < best_score
            return self.is_improved

    def _save(self, model: nn.Module, save_path: pathlib.Path) -> None:
        state = model.state_dict()
        torch.save(state, save_path)
        logger.info(f"Saved model {model.__class__.__name__}({type(model)}) to {save_path}")

    def check(self, score: float, model: nn.Module, save_path: pathlib.Path) -> None:
        if self._is_improved(score, self._best_score):
            logger.info(f"Score improved from {self._best_score} to {score}")
            self._best_score = score
            self._counter = 0
            self._save(model, save_path)
        else:
            self._counter += 1
            logger.info(
                f"EarlyStopping counter: {self._counter} out of {self._patience}. " + f"best: {self._best_score}"
            )

    @property
    def is_early_stop(self) -> bool:
        return self._counter >= self._patience

    @property
    def best_score(self) -> float:
        return self._best_score


class MetricsMonitor:
    def __init__(self, metrics: Sequence[str]) -> None:
        self.metrics = metrics
        self._metrics_df = pd.DataFrame(columns=[*metrics])  # type: ignore

    def update(self, metrics: dict[str, float | int]) -> None:
        epoch = cast(int, metrics.pop("epoch"))
        _metrics = pd.DataFrame(metrics, index=[epoch])  # type: ignore
        if self._metrics_df.empty:
            self._metrics_df = _metrics
        else:
            self._metrics_df = pd.concat([self._metrics_df, _metrics], axis=0)

        if wandb.run is not None:
            wandb.log(metrics)

    def show(self, log_interval: int = 1) -> None:
        loggin_metrics: pd.DataFrame = self._metrics_df.iloc[list(range(0, len(self._metrics_df), log_interval))]
        logger.info(f"\n{loggin_metrics.to_markdown()}")

    def plot(
        self,
        save_path: pathlib.Path,
        col: str | Sequence[str],
        figsize: tuple[int, int] = (8, 6),
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize)
        assert isinstance(ax, axes.Axes)
        assert isinstance(fig, figure.Figure)
        if isinstance(col, str):
            col = [col]
        for c in col:
            data = self._metrics_df[c].to_numpy()
            ax.plot(data, label=c)
        ax.set_xlabel("epoch")
        ax.set_ylabel(",".join(col))
        ax.legend()
        fig.savefig(save_path)

    def save(self, save_path: pathlib.Path, fold: int) -> None:
        self._metrics_df["fold"] = fold
        self._metrics_df.to_csv(save_path, index=False)


Optimizers: TypeAlias = torch_optim.Optimizer
Schdulers: TypeAlias = torch_optim.lr_scheduler.LRScheduler | timm_scheduler.cosine_lr.Scheduler


class TrainOneEpochFn(Protocol):
    def __call__(
        self,
        epoch: int,
        model: nn.Module,
        ema_model: timm_utils.ModelEmaV2 | None,
        dl_train: torch_data.DataLoader,
        optimizer: Optimizers,
        scheduler: Schdulers | None,
        loss_fn: my_losses.LossFn,
        scaler: torch.cuda.amp.grad_scaler.GradScaler | None,
        seed: int,
        use_amp: bool,
        device: torch.device,
    ) -> dict[str, float | int]: ...


class ValidOneEpochFn(Protocol):
    def __call__(
        self,
        epoch: int,
        model: nn.Module,
        ema_model: timm_utils.ModelEmaV2 | None,
        dl_valid: torch_data.DataLoader,
        loss_fn: my_losses.LossFn,
        seed: int,
        use_amp: bool,
        device: torch.device,
    ) -> dict[str, float | int]: ...


_T = TypeVar("_T", TrainOneEpochFn, ValidOneEpochFn)


def insert_duration_to_results(is_train: bool = True) -> Callable:
    mode = "train" if is_train else "valid"

    def _insert_duration_to_results(fn: _T) -> _T:
        @functools.wraps(fn)  # type: ignore
        def wrapper(*args, **kwargs):
            start_time = time.time()
            results = fn(*args, **kwargs)
            if not isinstance(results, dict):
                raise ValueError(f"Expected dict, but got {type(results)}")
            results[f"{mode}/duration"] = time.time() - start_time
            return results

        return cast(_T, wrapper)

    return _insert_duration_to_results


class TrainConfig(Protocol):
    n_epochs: int
    optimizer_name: my_optim.OptimizerType
    optimizer_params: my_optim.OptimParams
    scheduler_name: my_optim.SchdulerType
    scheduler_params: my_optim.SchduleParams
    loss_name: my_losses.LossType
    metric_names: Sequence[str]
    monitor_metric_name: str
    patience: int
    direction: EarlyStopDirection
    log_interval: int
    use_amp: bool
    dataloader_params: dict[str, Any]


class ValidConfig(Protocol):
    dataloader_params: dict[str, Any]


class Config(Protocol):
    name: str
    seed: int
    output_dir: pathlib.Path

    model_config: my_model_common.ModelConfig
    train_config: TrainConfig
    valid_config: ValidConfig


def train_one_fold(
    cfg: Config,
    fold: int,
    train_one_epoch: TrainOneEpochFn,
    dl_train: torch_data.DataLoader,
    valid_one_epoch: ValidOneEpochFn | None = None,
    dl_valid: torch_data.DataLoader | None = None,
    use_ema: bool = False,
    compile_config: my_model_common.CompileConfig | None = None,
) -> None:
    """1fold分の学習を行う
    Args:
        cfg: config
        fold: fold番号
        train_one_epoch: 1epoch分の学習を行う関数
        valid_one_epoch: 1epoch分のvalidationを行う関数
        use_ema: EMAを使うかどうか
        compile_config: モデルをコンパイルするかどうかの設定
    """
    if compile_config is None:
        compile_config = my_model_common.CompileConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = my_model_common.init_model(
        model_name=cfg.model_config.model_name, model_params=cfg.model_config.model_params
    )
    ema_model = timm_utils.ModelEmaV2(model, decay=0.998) if use_ema else None
    if compile_config.do_model_compile:
        logger.info("Compile model")
        model = my_model_common.compile_model(model, compile_config)
        if ema_model is not None:
            ema_model = my_model_common.compile_model(ema_model, compile_config)

    model = model.to(device, non_blocking=True)
    ema_model = ema_model.to(device, non_blocking=True) if ema_model is not None else None

    optimizer = my_optim.init_optimizer(cfg.train_config.optimizer_name, model, cfg.train_config.optimizer_params)
    scheduler = my_optim.init_scheduler(cfg.train_config.scheduler_name, optimizer, cfg.train_config.scheduler_params)
    scaler = torch.cuda.amp.grad_scaler.GradScaler() if cfg.train_config.use_amp else None
    loss_fn = my_losses.init_loss_fn(loss_name=cfg.train_config.loss_name)

    metrics_monitor = MetricsMonitor(metrics=cfg.train_config.metric_names)
    early_stopping = EarlyStopping(cfg.train_config.patience, cfg.train_config.direction)
    start_time = time.time()
    best_oof = pl.DataFrame()
    for epoch in range(cfg.train_config.n_epochs):
        my_common_utils.seed_everything(cfg.seed)
        logger.info(f"Start epoch: {epoch}")
        train_results = train_one_epoch(
            epoch=epoch,
            model=model,
            ema_model=ema_model,
            dl_train=dl_train,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            scaler=scaler,
            seed=cfg.seed,
            use_amp=cfg.train_config.use_amp,
            device=device,
        )
        if valid_one_epoch is not None and dl_valid is not None:
            valid_results = valid_one_epoch(
                epoch=epoch,
                model=model,
                ema_model=ema_model,
                dl_valid=dl_valid,
                loss_fn=loss_fn,
                seed=cfg.seed,
                use_amp=cfg.train_config.use_amp,
                device=device,
            )
            valid_oof = valid_results.pop("valid/oof")
            metrics_monitor.update({**train_results, **valid_results, "epoch": epoch})

            if epoch % cfg.train_config.log_interval == 0:
                metrics_monitor.show(log_interval=cfg.train_config.log_interval)

            score = valid_results[cfg.train_config.monitor_metric_name]
            model_save_path = cfg.output_dir / f"best_{cfg.name}_fold{fold}.pth"
            _save_model = model if not use_ema or ema_model is None else ema_model.module
            early_stopping.check(score=score, model=_save_model, save_path=model_save_path)
            if early_stopping.is_improved:
                best_oof = cast(pl.DataFrame, valid_oof)

            if early_stopping.is_early_stop:
                logger.info("Early stopping")
                break

    # -- end of epoch
    print(f"Best OOF : {best_oof}")
    best_oof.write_csv(cfg.output_dir / f"best_oof_fold{fold}.csv")
    metrics_monitor.save(save_path=cfg.output_dir / f"metrics_fold{fold}.csv", fold=fold)
    metrics_monitor.plot(save_path=cfg.output_dir / f"metrics_fold{fold}.png", col=["train/loss", "valid/loss"])
    state_dict = model.state_dict() if not use_ema or ema_model is None else ema_model.module.state_dict()
    fname = f"last_{cfg.name}_fold{fold}.pth" if valid_one_epoch is not None else f"full_{cfg.name}_fold{fold}.pth"
    torch.save(state_dict, cfg.output_dir / fname)

    elapsed_time = time.time() - start_time
    logger.info(
        f"Training fold{fold} is done. "
        f"elapsed time: {elapsed_time:.2f}[sec] "
        f"/ {elapsed_time / 60:.2f}[min] "
        f"/ {elapsed_time / 3600:.2f}[hour] "
    )
