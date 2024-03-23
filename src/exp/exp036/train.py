import argparse
import dataclasses
import pprint

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from timm import utils as timm_utils
from torch import backends
from torch.cuda import amp
from torch.utils import data as torch_data
from tqdm.auto import tqdm

import wandb
from src.exp.exp036 import config as my_config
from src.models import layers as my_layers
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


def make_specs_from_signals(
    signals: torch.Tensor,
    spec_module: torchaudio.transforms.Spectrogram | torchaudio.transforms.MelSpectrogram,
    db_transform: torchaudio.transforms.AmplitudeToDB | None = None,
) -> torch.Tensor:
    bs = signals.size(0)
    # signal to specs
    # signal shape: (bs, 4, 4, 10000)
    specs = torch.zeros((bs, 4, 128, 256))
    for k in range(4):
        for kk in range(4):
            # spec shape: (bs, 129, 257)
            spec = spec_module(signals[:, k, kk, :])
            if db_transform is not None:
                spec = db_transform(spec)

            height = (spec.shape[1] // 32) * 32
            width = (spec.shape[2] // 32) * 32
            spec = spec[:, :height, :width]
            # Standardization -1 to 1
            # mel_spec shape: (bs, 4, 128, 256)

            specs[:, k, :, :] += spec
            specs[:, k, :, :] /= 4

    # specs = (specs - specs.mean()) / (specs.std() + 1e-6)
    # specs = (specs - specs.min()) / (specs.max() - specs.min() + 1e-6)
    return specs


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
    # -- Freeze/Unfreeze
    keys = ["backbone"]
    if epoch < 3:
        my_tools.freeze(model, keys)
    else:
        my_tools.unfreeze(model, keys)

    spec_module = torchaudio.transforms.MelSpectrogram(
        sample_rate=200,
        n_fft=1024,
        hop_length=10000 // 256,
        win_length=128,
        n_mels=128,
        f_min=0,
        f_max=20,
        power=2.0,
        pad_mode="constant",
        center=True,
    )
    db_transform = my_layers.init_db_transform_module()
    # Spectrogram module
    time_mask_size = np.random.randint(1, 256)
    freq_mask_size = np.random.randint(1, 128)
    spec_aug = nn.Sequential(
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15, iid_masks=True),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_size, iid_masks=True),
        torchaudio.transforms.TimeMasking(time_mask_param=35, iid_masks=False, p=0.5),
        torchaudio.transforms.TimeMasking(time_mask_param=time_mask_size, iid_masks=True, p=0.5),
    )

    # -- Mixup
    mixupper = my_preprocessings.Mixup(p=0.75, alpha=0.5)
    if epoch < 13:
        mixupper.init_lambda()

    # -- Train Loop
    pbar = tqdm(enumerate(dl_train), total=len(dl_train), desc=f"Train {epoch}")
    for i, batch in pbar:
        my_utils_common.seed_everything(seed)
        if ema_model is not None:
            ema_model.update(model)

        y = batch["y"].to(device, non_blocking=True, dtype=torch.float32)
        with amp.autocast_mode.autocast(enabled=use_amp):
            # shape x: (bs, 128, 256, 8) at dim=3, 0~3: specs_from_eeg, 4~7: host_specs
            x = batch["x"].to(device, non_blocking=True, dtype=torch.float32)

            # -- Signal To MelSpectrograms
            specs = make_specs_from_signals(signals=batch["signals"], spec_module=spec_module)
            specs = db_transform(specs)
            specs = spec_aug(specs)

            # clip outliers for stability and standardization
            specs = torch.clamp(specs, min=np.exp(-4.0), max=np.exp(8.0))
            specs = torch.log(specs)
            specs = (specs - specs.mean()) / (specs.std() + 1e-6)

            specs = specs.permute(0, 2, 3, 1)
            specs = specs.to(device, non_blocking=True, dtype=torch.float32)

            # x shape: (bs, 128, 256, 12)
            x = torch.cat([x, specs], dim=3)

            if mixupper.do_mixup:
                # sample内でのmixup
                x = mixupper.lam * x + (1 - mixupper.lam) * x.flip(3)

                # sample間でのmixup
                x = mixupper.lam * x + (1 - mixupper.lam) * x.flip(0)
                y = mixupper.lam * y + (1 - mixupper.lam) * y.flip(0)

            bboxes, k_list, a_list = my_preprocessings.prepare_cutmix(x_max=512, y_max=512, bs=x.size(0))
            out = model(
                x,
                do_mixup=mixupper.do_mixup,
                lam=mixupper.lam,
                bboxes=bboxes,
                k_list=k_list,
                a_list=a_list,
                y=y,
            )

        logits = out["logits"]
        if mixupper.do_mixup:
            mixed_loss = loss_fn(
                F.log_softmax(logits, dim=1), out["y"].to(device, non_blocking=True, dtype=logits.dtype)
            )
            normal_loss = loss_fn(F.log_softmax(logits, dim=1), y.flip(0))

            loss = mixupper.lam * mixed_loss + (1 - mixupper.lam) * normal_loss

        else:
            loss = loss_fn(F.log_softmax(logits, dim=1), y)

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

    train_results = {"train/loss": losses.avg}
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

    # Spectrogram module
    spec_module = torchaudio.transforms.MelSpectrogram(
        sample_rate=200,
        n_fft=1024,
        hop_length=10000 // 256,
        win_length=128,
        n_mels=128,
        f_min=0,
        f_max=20,
        power=2.0,
        pad_mode="constant",
        center=True,
    )
    db_transform = my_layers.init_db_transform_module()

    spec_offsets = []
    eeg_ids_list = []
    spec_ids_list = []
    y_trues_list = []
    y_preds_list = []
    y_raws_list = []
    pbar = tqdm(enumerate(dl_valid), total=len(dl_valid), desc=f"Valid {epoch}")
    for _, batch in pbar:
        my_utils_common.seed_everything(seed)

        y = batch["y"].to(device, non_blocking=True, dtype=torch.float32)
        with amp.autocast_mode.autocast(enabled=use_amp), torch.inference_mode():
            x = batch["x"].to(device, non_blocking=True, dtype=torch.float32)

            # -- Signal To MelSpectrograms
            specs = make_specs_from_signals(signals=batch["signals"], spec_module=spec_module)
            specs = db_transform(specs)

            # clip outliers for stability and standardization
            specs = torch.clamp(specs, min=np.exp(-4.0), max=np.exp(8.0))
            specs = torch.log(specs)
            specs = (specs - specs.mean()) / (specs.std() + 1e-6)

            specs = specs.permute(0, 2, 3, 1)
            specs = specs.to(device, non_blocking=True, dtype=torch.float32)

            # x shape: (bs, 128, 256, 12)
            x = torch.cat([x, specs], dim=3)

            out = model(x)

        logits = out["logits"]
        loss = loss_fn(F.log_softmax(logits, dim=1), y).detach().cpu()
        losses.update(loss.item())

        y_pred = F.softmax(logits, dim=1)

        spec_offsets.append(batch["spec_offset"])
        eeg_ids_list.append(batch["eeg_id"])
        spec_ids_list.append(batch["spec_id"])
        y_preds_list.append(y_pred.cpu().detach().numpy())
        y_trues_list.append(y.cpu().detach().numpy())
        y_raws_list.append(batch["y_raw"].cpu().detach().numpy())

        pbar.set_postfix({"loss": losses.avg, "acc": accs.avg})

    y_trues = np.concatenate(y_trues_list, axis=0)
    y_preds = np.concatenate(y_preds_list, axis=0)
    acc = (y_preds.argmax(1) == y_trues.argmax(1)).mean()
    y_raws = np.concatenate(y_raws_list, axis=0)
    valid_oof = pl.DataFrame({
        "eeg_id": np.concatenate(eeg_ids_list, axis=0),
        "spec_id": np.concatenate(spec_ids_list, axis=0),
        "spec_offset": np.concatenate(spec_offsets, axis=0),
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
        "valid/acc": acc,
    }
    return valid_results


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
                project="kaggle-hms-2024", name=f"{cfg.name}_fold{fold}", config=dataclasses.asdict(cfg), reinit=True
            )
        else:
            run = None

        my_tools.train_one_fold(
            cfg=cfg,
            fold=fold,
            train_one_epoch=train_one_epoch,
            dl_train=my_data.init_train_dataloader(fold, **cfg.train_config.dataloader_params),
            valid_one_epoch=valid_one_epoch,
            dl_valid=my_data.init_valid_dataloader(fold, **cfg.valid_config.dataloader_params),
            use_ema=True,
        )
        if run is not None:
            run.finish()
        logger.info(f"Finished fold={fold}")
    logger.info("Finished all training")


if __name__ == "__main__":
    main()
