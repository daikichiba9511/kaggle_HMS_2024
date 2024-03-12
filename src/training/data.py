from logging import getLogger

import albumentations as A
import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils import data as torch_data

from src import constants
from src.training import cv as my_cv
from src.training import preprocessings as my_preprocessings
from src.utils import common as my_utils_common

logger = getLogger(__name__)


class TrainHMSDataset(torch_data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        spectrograms: dict[str, np.ndarray],
        eeg_specs: dict[str, np.ndarray],
        transform: A.Compose | None = None,
        mode: str = "tta",
    ) -> None:
        self.df = df
        self.spectrograms = spectrograms
        self.eeg_spectrograms = eeg_specs
        self.transform = transform
        self.mode = mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        x = np.zeros((128, 256, 8), dtype=np.float32)
        y = np.zeros(6, dtype=np.float32)
        img = np.ones((128, 256), dtype=np.float32)
        row = self.df.iloc[idx]
        # r = int(row["spectrogram_label_offset_seconds"]) // 2
        r = int((row["min"] + row["max"]) // 4)

        for region in range(4):
            img = self.spectrograms[row["spec_id"]][r : r + 300, region * 100 : (region + 1) * 100].T
            # Log transform spectrogram
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # Standarize per image
            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)
            x[14:-14, :, region] = img[:, 22:-22] / 2.0
            img = self.eeg_spectrograms[row["eeg_id"]]
            x[:, :, 4:] = img

        y = row[constants.TARGETS].to_numpy().astype(np.float32)
        y += 1 / 6
        y /= y.sum()

        if self.transform is not None:
            x = self.transform(image=x)["image"]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return {
            "x": x,
            "y": y,
            "eeg_id": row["eeg_id"],
            "spec_id": row["spec_id"],
        }


class ValidHMSDataset(torch_data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        spectrograms: dict[str, np.ndarray],
        eeg_specs: dict[str, np.ndarray],
        transform: A.Compose | None = None,
        mode: str = "tta",
    ) -> None:
        self.df = df
        self.spectrograms = spectrograms
        self.eeg_spectrograms = eeg_specs
        self.transform = transform
        self.mode = mode

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        x = np.zeros((128, 256, 8), dtype=np.float32)
        y = np.zeros(6, dtype=np.float32)
        img = np.ones((128, 256), dtype=np.float32)
        row = self.df.iloc[idx]
        # r = int(row["spectrogram_label_offset_seconds"]) // 2
        r = int((row["min"] + row["max"]) // 4)

        for region in range(4):
            img = self.spectrograms[row["spec_id"]][r : r + 300, region * 100 : (region + 1) * 100].T
            # Log transform spectrogram
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # Standarize per image
            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)

            x[14:-14, :, region] = img[:, 22:-22] / 2.0
            img = self.eeg_spectrograms[row["eeg_id"]]
            x[:, :, 4:] = img

        y = row[constants.TARGETS].to_numpy().astype(np.float32)
        y += 1 / 6
        y /= y.sum()

        if self.transform is not None:
            x = self.transform(image=x)["image"]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        y_raw = torch.tensor(row[constants.TARGETS].to_numpy().astype(np.float32), dtype=torch.float32)

        return {
            "x": x,
            "y": y,
            "y_raw": y_raw,
            "eeg_id": row["eeg_id"],
            "spec_id": row["spec_id"],
            # "spec_offset": row["spectrogram_label_offset_seconds"],
            "spec_offset": r * 2,
        }


_DF_CACHE: pl.DataFrame | None = None


def _set_df_cache(df: pl.DataFrame) -> None:
    logger.info("Setting df cache")
    print("Setting df cache: print")
    global _DF_CACHE
    _DF_CACHE = df


def _load_df() -> pl.DataFrame:
    if isinstance(_DF_CACHE, pl.DataFrame):
        logger.info("Using df cache")
        print("Setting df cache: print")
        return _DF_CACHE

    # Ref:
    # 1. scripts/eda/eda002.py
    # 2. https://www.kaggle.com/code/alejopaullier/hms-efficientnetb0-pytorch-train
    # train_df = pl.read_csv(constants.DATA_DIR / "train.csv")
    train_df_fp = constants.DATA_DIR / "train_preprocessed.csv"
    train_df = pl.read_csv(train_df_fp)
    logger.info(f"{train_df.shape = } {train_df_fp = }")
    logger.info(f"{train_df = }")
    train_df = my_cv.make_fold(df=train_df, group_col_name="patient_id", target_col_name="target")

    # common preprocessing for all df
    train_df = train_df.rename({"spectrogram_id": "spec_id"})
    train_df = train_df.with_columns([
        pl.col("spec_id").cast(pl.Utf8),
        pl.col("eeg_id").cast(pl.Utf8),
    ])

    _set_df_cache(train_df)
    return train_df


def load_train_df(fold: int) -> pl.DataFrame:
    df = _load_df()
    train_df = df.filter(pl.col("fold") != fold)
    return train_df


def load_valid_df(fold: int) -> pl.DataFrame:
    df = _load_df()
    valid_df = df.filter(pl.col("fold") == fold)
    return valid_df


def _make_specs(df: pl.DataFrame) -> dict[str, np.ndarray]:
    # spectrograms is made from eeg
    # eeg_id -> spectrogram
    eeg_ids = df["eeg_id"].unique().to_list()
    spectrograms_from_eeg = my_preprocessings.make_spectrograms_from_eeg(eeg_ids=eeg_ids, with_progress_bar=True)
    print("intersection of spectrograms & eeg_id", len(set(spectrograms_from_eeg.keys()) & set(df["eeg_id"])))
    return spectrograms_from_eeg


_SPECS_CACHE: dict[str, np.ndarray] | None = None


def _load_specs_from_eeg() -> dict[str, np.ndarray]:
    global _SPECS_CACHE
    if isinstance(_SPECS_CACHE, dict):
        logger.info("Use cache spectrogram from eeg")
        return _SPECS_CACHE
    specs = np.load(constants.OUTPUT_DIR / "spectrograms_from_eeg.npy", allow_pickle=True).item()
    _SPECS_CACHE = specs
    return specs


def init_train_dataloader(
    fold: int, batch_size: int = 8 * 3, num_workers: int = 4, is_debug: bool = False, remake_specs: bool = False
) -> torch_data.DataLoader:
    train_df = load_train_df(fold)
    # spectrograms was prepared by the host
    # spec_id -> spectrogram
    host_specs = my_preprocessings.load_spectrograms(with_progress_bar=True, is_debug=is_debug)
    if is_debug:
        print(f"Before filtering: {train_df.shape}")
        train_df = train_df.filter(pl.col("spec_id").is_in(host_specs.keys()))
        print(f"After filtering: {train_df.shape}")
        print(train_df)

    if remake_specs:
        spectrograms_from_eeg = _make_specs(train_df)
    else:
        spectrograms_from_eeg = _load_specs_from_eeg()

    ds = TrainHMSDataset(
        train_df.to_pandas(use_pyarrow_extension_array=True),
        spectrograms=host_specs,
        eeg_specs=spectrograms_from_eeg,
        transform=my_preprocessings.get_transforms(is_train=True),
        mode="tta",
    )

    dl = torch_data.DataLoader(
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


def init_valid_dataloader(
    fold: int, batch_size: int = 8 * 3, num_workers: int = 4, is_debug: bool = False, remake_specs: bool = False
) -> torch_data.DataLoader:
    valid_df = load_valid_df(fold)
    if is_debug:
        spec_ids = valid_df["spec_id"].to_list()[:20]
        fps = [fp for fp in constants.DATA_DIR.joinpath("train_spectrograms").glob("*.parquet") if fp.stem in spec_ids]
        print(f"{len(fps) = }")
        valid_df = valid_df.filter(pl.col("spec_id").is_in([fp.stem for fp in fps]))
        host_specs = my_preprocessings.load_spectrograms(with_progress_bar=True, is_debug=is_debug, fps=fps)
    else:
        # spectrograms was prepared by the host
        # spec_id -> spectrogram
        host_specs = my_preprocessings.load_spectrograms(with_progress_bar=True, is_debug=is_debug)

    print(f"host_specs: {len(host_specs) = }")

    if remake_specs:
        spectrograms_from_eeg = _make_specs(valid_df)
    else:
        spectrograms_from_eeg = _load_specs_from_eeg()

    ds = ValidHMSDataset(
        valid_df.to_pandas(use_pyarrow_extension_array=True),
        spectrograms=host_specs,
        eeg_specs=spectrograms_from_eeg,
        # transform=my_preprocessings.get_transforms(is_train=True),
        transform=None,
        mode="tta",
    )

    dl = torch_data.DataLoader(
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


def _test_train_dl() -> None:
    dl = init_train_dataloader(fold=0, is_debug=True)
    batch = next(iter(dl))
    print(batch["x"].shape)
    print(batch["y"].shape)
    print(batch["y"])
    print(f"{batch['y'].min() = }, {batch['y'].max() = }")


def _test() -> None:
    _test_train_dl()


if __name__ == "__main__":
    _test()
