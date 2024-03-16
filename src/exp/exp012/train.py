import argparse
import dataclasses
import pprint

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import utils as timm_utils
from torch import backends
from torch.cuda import amp
from torch.utils import data as torch_data
from tqdm.auto import tqdm

import wandb
from src import constants
from src.exp.exp012 import config as my_config
from src.training import data as my_data
from src.training import losses as my_losses
from src.training import preprocessings as my_preprocessings
from src.training import tools as my_tools
from src.utils import common as my_utils_common
from src.utils import logger as my_logger

logger = my_logger.get_root_logger()


backends.cuda.matmul.allow_tf32 = True
backends.cudnn.allow_tf32 = True
backends.cudnn.benchmark = False
backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--train-folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    return parser.parse_args()


@my_tools.insert_duration_to_results(is_train=True)
def train_one_epoch(
    epoch: int,
    model: nn.Module,
    ema_model: timm_utils.ModelEmaV2 | None,
    dl_train: torch_data.DataLoader,
    optimizer: my_tools.Optimizers,
    scheduler: my_tools.Schdulers | None,
    loss_fn: my_losses.LossFn,
    scaler: torch.cuda.amp.grad_scaler.GradScaler | None,
    seed: int,
    use_amp: bool,
    device: torch.device,
    # -- additional arguments --
    grad_accum_steps: int = 1,
) -> dict[str, float]:
    model = model.train().to(device, non_blocking=True)

    if scheduler is None:
        raise ValueError("scheduler is None")
    scheduler.step(epoch=epoch)
    losses = my_tools.AverageMeter("losses")

    # -- Mixup
    mixupper = my_preprocessings.Mixup(p=0.5, alpha=0.5)
    mixupper.init_lambda()
    pbar = tqdm(enumerate(dl_train), total=len(dl_train), desc=f"Train {epoch}")
    for i, batch in pbar:
        my_utils_common.seed_everything(seed)
        if ema_model is not None:
            ema_model.update(model)

        y = batch["y"].to(device, non_blocking=True, dtype=torch.float32)
        with amp.autocast_mode.autocast(enabled=use_amp):
            x = batch["x"].to(device, non_blocking=True, dtype=torch.float32)
            if mixupper.do_mixup:
                x = mixupper.lam * x + (1.0 - mixupper.lam) * x.flip(0)
            out = model(x)

        logits = out["logits"]
        loss = loss_fn(F.log_softmax(logits, dim=1), y)
        if mixupper.do_mixup:
            loss *= mixupper.lam
            loss += (1.0 - mixupper.lam) * loss_fn(F.log_softmax(logits, dim=1), y.flip(0).to(dtype=logits.dtype))

        if i % grad_accum_steps == 0:
            my_tools.backward_step(
                batch_idx=i,
                loss=loss,
                optimizer=optimizer,
                scaler=scaler,
                model=model,
                grad_accum_steps=grad_accum_steps,
            )
            losses.update(loss.item())
            pbar.set_postfix({"loss": losses.avg})

    train_results = {"train/loss": losses.avg, "lr": optimizer.param_groups[0]["lr"]}
    return train_results


@my_tools.insert_duration_to_results(is_train=False)
def valid_one_epoch(
    epoch: int,
    model: nn.Module,
    ema_model: timm_utils.ModelEmaV2 | None,
    dl_valid: torch_data.DataLoader,
    loss_fn: my_losses.LossFn,
    seed: int,
    use_amp: bool,
    device: torch.device,
) -> dict[str, float]:
    model = model.eval().to(device)
    losses = my_tools.AverageMeter("losses")
    accs = my_tools.AverageMeter("accs")
    pbar = tqdm(enumerate(dl_valid), total=len(dl_valid), desc=f"Valid {epoch}")

    # spec_offsets = []
    eeg_ids_list = []
    spec_ids_list = []
    y_trues_list = []
    y_preds_list = []
    y_raws_list = []
    for _, batch in pbar:
        my_utils_common.seed_everything(seed)

        y = batch["y"].to(device, non_blocking=True, dtype=torch.float32)
        with amp.autocast_mode.autocast(enabled=use_amp), torch.inference_mode():
            x = batch["x"].to(device, non_blocking=True, dtype=torch.float32)
            out = model(x)

        logits = out["logits"]
        loss = loss_fn(F.log_softmax(logits, dim=1), y).detach().cpu()
        losses.update(loss.item())

        y_pred = F.softmax(logits, dim=1)

        # spec_offsets.append(batch["spec_offset"])
        eeg_ids_list.append(batch["eeg_id"])
        spec_ids_list.append(batch["spec_id"])
        y_preds_list.append(y_pred.cpu().detach().numpy())
        y_trues_list.append(y.cpu().detach().numpy())
        y_raws_list.append(batch["y_raw"].cpu().detach().numpy())

        pbar.set_postfix({"loss": losses.avg, "acc": accs.avg})

    y_trues = np.concatenate(y_trues_list, axis=0)
    y_preds = np.concatenate(y_preds_list, axis=0)
    y_raws = np.concatenate(y_raws_list, axis=0)
    valid_oof = pl.DataFrame({
        "eeg_id": np.concatenate(eeg_ids_list, axis=0),
        "spec_id": np.concatenate(spec_ids_list, axis=0),
        # "spec_offset": np.concatenate(spec_offsets, axis=0),
        # -- labels --
        "label_seizure_vote": y_trues[:, 0],
        "label_lpd_vote": y_trues[:, 1],
        "label_gpd_vote": y_trues[:, 2],
        "label_lrda_vote": y_trues[:, 3],
        "label_grda_vote": y_trues[:, 4],
        "label_other_vote": y_trues[:, 5],
        # -- ray labels --
        "raw_seizure_vote": y_raws[:, 0],
        "raw_lpd_vote": y_raws[:, 1],
        "raw_gpd_vote": y_raws[:, 2],
        "raw_lrda_vote": y_raws[:, 3],
        "raw_grda_vote": y_raws[:, 4],
        "raw_other_vote": y_raws[:, 5],
        # -- preds --
        "predict_seizure_vote": y_preds[:, 0],
        "predict_lpd_vote": y_preds[:, 1],
        "predict_gpd_vote": y_preds[:, 2],
        "predict_lrda_vote": y_preds[:, 3],
        "predict_grda_vote": y_preds[:, 4],
        "predict_other_vote": y_preds[:, 5],
    })
    valid_results = {
        "valid/loss": losses.avg,
        "valid/oof": valid_oof,
    }
    return valid_results


def _init_train_dataloader(
    fold: int,
    batch_size: int = 8 * 3,
    num_workers: int = 4,
    is_debug: bool = False,
    remake_specs: bool = False,
    reduce_noise: bool = True,
) -> torch_data.DataLoader[my_data.TrainEEGOutput]:
    df = my_data.load_train_df(fold=fold)
    eegs = my_preprocessings.retrieve_eegs_from_parquet(df["eeg_id"].unique().to_list(), constants.feature_cols)
    ds = my_data.TrainEEGDataset(
        df=df.to_pandas(use_pyarrow_extension_array=True),
        eegs=eegs,
        transform=None,
        reduce_noise=reduce_noise,
    )
    dl: torch_data.DataLoader[my_data.TrainEEGOutput] = torch_data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=lambda _: my_utils_common.seed_everything(42),
        persistent_workers=True,
    )
    return dl


def _init_valid_dataloader(
    fold: int,
    batch_size: int = 8 * 3,
    num_workers: int = 4,
    is_debug: bool = False,
    remake_specs: bool = False,
    reduce_noise: bool = True,
) -> torch_data.DataLoader[my_data.ValidEEGOutput]:
    df = my_data.load_valid_df(fold=fold)
    eegs = my_preprocessings.retrieve_eegs_from_parquet(df["eeg_id"].unique().to_list(), constants.feature_cols)
    ds = my_data.ValidEEGDataset(
        df=df.to_pandas(use_pyarrow_extension_array=True),
        eegs=eegs,
        transform=None,
        reduce_noise=reduce_noise,
    )
    dl: torch_data.DataLoader[my_data.ValidEEGOutput] = torch_data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=lambda _: my_utils_common.seed_everything(42),
        persistent_workers=True,
    )
    return dl


def main() -> None:
    args = parse_args()
    cfg = my_config.ConfigImpl()
    if args.debug:
        cfg.train_config.dataloader_params["is_debug"] = True
        cfg.valid_config.dataloader_params["is_debug"] = True
        args.train_folds = [0]

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Config: {pprint.pformat(dataclasses.asdict(cfg))}")
    my_logger.attach_file_handler(logger, str(cfg.output_dir / "train.log"))

    for fold in range(5):
        if fold not in args.train_folds:
            logger.info(f"Skip fold={fold}, since it's not in args.train_folds {args.train_folds}")
            continue
        my_utils_common.seed_everything(cfg.seed)

        if not args.debug:
            run = wandb.init(
                project="kaggle-hms-2024",
                name=f"{cfg.name}_fold{fold}",
                config=dataclasses.asdict(cfg),
                reinit=True,
                group=cfg.name,
                dir="./src",
            )
        else:
            run = None

        my_tools.train_one_fold(
            cfg=cfg,
            fold=fold,
            train_one_epoch=train_one_epoch,
            # dl_train=my_data.init_train_dataloader(fold, **cfg.train_config.dataloader_params),
            dl_train=_init_train_dataloader(fold, **cfg.train_config.dataloader_params),
            valid_one_epoch=valid_one_epoch,
            # dl_valid=my_data.init_valid_dataloader(fold, **cfg.valid_config.dataloader_params),
            dl_valid=_init_valid_dataloader(fold, **cfg.valid_config.dataloader_params),
            use_ema=True,
        )
        if run is not None:
            run.finish()
        logger.info(f"Finished fold={fold}")
    logger.info("Finished all training")


if __name__ == "__main__":
    main()
