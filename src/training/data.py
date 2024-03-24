from logging import getLogger
from typing import TypedDict

import albumentations as A
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import torch
from torch.utils import data as torch_data
from typing_extensions import Self

from src import constants
from src.training import cv as my_cv
from src.training import preprocessings as my_preprocessings
from src.utils import common as my_utils_common

logger = getLogger(__name__)


def _consolidate_patches_into_a_image(x: torch.Tensor) -> torch.Tensor:
    """make a spectrogram from the eeg and the spectrogram

    Args:
        x (torch.Tensor): input tensor, shape (bs, 128, 256, 8)
    """
    # shape: (bs, 4, 128, 256)
    # spectrograms = [x[:, i : i + 1, :, :] for i in range(4)]
    spectrograms = [x[:, :, :, i : i + 1] for i in range(4)]
    spectrograms = torch.cat(spectrograms, dim=1)

    # shape: (bs, 4, 128, 256)
    eegs = [x[:, :, :, i : i + 1] for i in range(4, 8)]
    eegs = torch.cat(eegs, dim=1)

    x = torch.cat([spectrograms, eegs], dim=2)
    img = torch.cat([x, x, x], dim=3)

    # region x {spectrogram, eeg} のスペクトログラムの貼り合わせ x 3 (channel direction)
    # |-------------------|
    # |    |    |    |    |
    # |-------------------|
    # |    |    |    |    |
    # |-------------------|
    #
    # permute to (bs, 3, 512, 512) to order channel first
    # shape: (bs, 512, 512, 3) -> (bs, 3, 512, 512)
    img = img.permute(0, 3, 1, 2)
    return img


class TrainHMSDataset(torch_data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        spectrograms: dict[str, npt.NDArray[np.float32]],
        eeg_specs: dict[str, npt.NDArray[np.float32]],
        transform: A.Compose | None = None,
        signals: dict[str, npt.NDArray[np.float32]] | None = None,
        random_crop_signal_augmentation: bool = False,
    ) -> None:
        self.df = df
        self.spectrograms = spectrograms
        self.eeg_spectrograms = eeg_specs
        self.transform = transform
        self.signals = signals

    def __len__(self: Self) -> int:
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
        yraw = torch.tensor(row[constants.TARGETS].to_numpy().astype(np.float32), dtype=torch.float32)

        if self.signals is not None:
            # shape: (4, 4, 10000)
            signals = self.signals[row["eeg_id"]]
            # signals = signals.mean(axis=1).reshape(4, -1)
            signals = torch.tensor(signals, dtype=torch.float32)
        else:
            signals = torch.tensor(0)

        return {
            "x": x,
            "y": y,
            "eeg_id": row["eeg_id"],
            "spec_id": row["spec_id"],
            "signals": signals,
            "y_raw": yraw,
        }


class ValidHMSDataset(torch_data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        spectrograms: dict[str, np.ndarray],
        eeg_specs: dict[str, np.ndarray],
        transform: A.Compose | None = None,
        signals: dict[str, npt.NDArray[np.float32]] | None = None,
    ) -> None:
        self.df = df
        self.spectrograms = spectrograms
        self.eeg_spectrograms = eeg_specs
        self.transform = transform
        self.signals = signals

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
        if self.signals is not None:
            signals = self.signals[row["eeg_id"]]
            # signals = signals.mean(axis=1).reshape(4, -1)
            signals = torch.tensor(signals, dtype=torch.float32)
        else:
            signals = torch.tensor(0)

        return {
            "x": x,
            "y": y,
            "y_raw": y_raw,
            "eeg_id": row["eeg_id"],
            "spec_id": row["spec_id"],
            # "spec_offset": row["spectrogram_label_offset_seconds"],
            "spec_offset": r * 2,
            "signals": signals,
        }


def _preprocess_raw_signal_for_eeg_dataset(
    data: npt.NDArray[np.floating], reduce_noise: bool = True, channel_normalize: bool = True
) -> npt.NDArray[np.floating]:
    # exp007,008
    # data = np.clip(data, -1024, 1024)
    # data = np.nan_to_num(data, nan=0.0) / 32.0

    # exp009
    data = np.clip(data, np.exp(-4), np.exp(8))
    data = np.log(data)
    data = np.nan_to_num(data, nan=0.0)

    # to remove the noise and high frequency signals
    data = my_preprocessings.preprocess_raw_signal(
        data, classes=1, reduce_noise=reduce_noise, channel_normalize=channel_normalize
    )
    return data


class TrainEEGOutput(TypedDict):
    x: torch.Tensor
    """shape: (10_000, 20), cols: see constants.feature_cols
    """
    y: torch.Tensor
    """shape: (6,),
        smoothed value in [0, 1].
        order: see constants.TARGETS
    """
    y_raw: torch.Tensor
    eeg_id: str
    spec_id: str


class TrainEEGDataset(torch_data.Dataset):
    """
    Ref:
    1. https://www.kaggle.com/code/nischaydnk/lightning-1d-eegnet-training-pipeline-hbs?scriptVersionId=160814948
    """

    def __init__(
        self: Self,
        df: pd.DataFrame,
        eegs: dict[str, npt.NDArray[np.float32]],
        transform: A.Compose | None = None,
        reduce_noise: bool = True,
        channel_normalize: bool = True,
    ) -> None:
        self.df = df
        self.eegs = eegs
        self.transform = transform
        self.reduce_noise = reduce_noise
        self.channel_normalize = channel_normalize

    def __len__(self: Self) -> int:
        return len(self.df)

    def __getitem__(self: Self, idx: int) -> TrainEEGOutput:
        row = self.df.iloc[idx]
        data = self.eegs[row["eeg_id"]]
        data = _preprocess_raw_signal_for_eeg_dataset(data, self.reduce_noise, self.channel_normalize)
        x = torch.tensor(data, dtype=torch.float32)
        # x = x.permute(1, 0)

        if self.transform is not None:
            x = self.transform(image=x)["image"]

        y = row[constants.TARGETS].to_numpy().astype(np.float32)
        y += 1 / 6
        y /= y.sum()

        y = torch.tensor(y, dtype=torch.float32)
        y_raw = torch.tensor(row[constants.TARGETS].to_numpy().astype(np.float32), dtype=torch.float32)

        return {
            "x": x,
            "y": y,
            "y_raw": y_raw,
            "eeg_id": row["eeg_id"],
            "spec_id": row["spec_id"],
        }


class ValidEEGOutput(TypedDict):
    x: torch.Tensor
    """shape: (10_000, 20), cols: see constants.feature_cols
    """
    y: torch.Tensor
    """shape: (6,),
        smoothed value in [0, 1].
        order: see constants.TARGETS
    """
    y_raw: torch.Tensor
    """shape: (6,), raw value of the count by annotator.
    """
    eeg_id: str
    spec_id: str


class ValidEEGDataset(torch_data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        eegs: dict[str, npt.NDArray[np.float32]],
        transform: A.Compose | None = None,
        reduce_noise: bool = True,
        channel_normalize: bool = True,
    ) -> None:
        self.df = df
        self.eegs = eegs
        self.transform = transform
        self.reduce_noise = reduce_noise
        self.channel_normalize = channel_normalize

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> ValidEEGOutput:
        row = self.df.iloc[idx]
        data = self.eegs[row["eeg_id"]]
        data = _preprocess_raw_signal_for_eeg_dataset(data, self.reduce_noise, self.channel_normalize)
        x = torch.tensor(data, dtype=torch.float32)
        # x = x.permute(1, 0)

        y = row[constants.TARGETS].to_numpy().astype(np.float32)
        y += 1 / 6
        y /= y.sum()

        y_raw = torch.tensor(row[constants.TARGETS].to_numpy().astype(np.float32), dtype=torch.float32)

        return {
            "x": x,
            "y": y,
            "y_raw": y_raw,
            "eeg_id": row["eeg_id"],
            "spec_id": row["spec_id"],
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

    train_df = train_df.with_columns(
        (
            pl.col("seizure_vote")
            + pl.col("lpd_vote")
            + pl.col("gpd_vote")
            + pl.col("lrda_vote")
            + pl.col("grda_vote")
            + pl.col("other_vote")
        ).alias("total_evaluators")
    )

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


def _make_specs(df: pl.DataFrame) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, npt.NDArray[np.float32]]]:
    # spectrograms is made from eeg
    # eeg_id -> spectrogram
    eeg_ids = df["eeg_id"].unique().to_list()
    spectrograms_from_eeg, signals = my_preprocessings.make_spectrograms_from_eeg(
        eeg_ids=eeg_ids, with_progress_bar=True
    )
    print("intersection of spectrograms & eeg_id", len(set(spectrograms_from_eeg.keys()) & set(df["eeg_id"])))
    return spectrograms_from_eeg, signals


_SPECS_CACHE: dict[str, np.ndarray] | None = None


def _load_specs_from_eeg() -> dict[str, np.ndarray]:
    global _SPECS_CACHE
    if isinstance(_SPECS_CACHE, dict):
        logger.info("Use cache spectrogram from eeg")
        return _SPECS_CACHE
    specs = np.load(constants.OUTPUT_DIR / "spectrograms_from_eeg.npy", allow_pickle=True).item()
    _SPECS_CACHE = specs
    return specs


_SIGNALS_CACHE: dict[str, npt.NDArray[np.float32]] | None = None


def _load_signals_from_eeg() -> dict[str, np.ndarray]:
    global _SIGNALS_CACHE
    if isinstance(_SIGNALS_CACHE, dict):
        logger.info("Use cache spectrogram from eeg")
        return _SIGNALS_CACHE
    # shape: eeg_id -> signal: (4, 4, 10000)
    signals = np.load(constants.OUTPUT_DIR / "signals_from_eeg.npy", allow_pickle=True).item()
    _SIGNALS_CACHE = signals
    return signals


def init_train_dataloader(
    fold: int,
    batch_size: int = 8 * 3,
    num_workers: int = 4,
    is_debug: bool = False,
    remake_specs: bool = False,
    step1: bool = False,
    step2: bool = False,
) -> torch_data.DataLoader[TrainEEGOutput]:
    train_df = load_train_df(fold)
    if step1:
        train_df = my_preprocessings.filter_less_df_by_votes(train_df)
    elif step2:
        train_df = my_preprocessings.filter_more_df_by_votes(train_df)

    # spectrograms was prepared by the host
    # spec_id -> spectrogram
    host_specs = my_preprocessings.load_spectrograms(with_progress_bar=True, is_debug=is_debug)
    if is_debug:
        print(f"Before filtering: {train_df.shape}")
        train_df = train_df.filter(pl.col("spec_id").is_in(host_specs.keys()))
        print(f"After filtering: {train_df.shape}")
        print(train_df)

    if remake_specs:
        spectrograms_from_eeg, signals_from_eeg = _make_specs(train_df)
    else:
        spectrograms_from_eeg = _load_specs_from_eeg()
        signals_from_eeg = _load_signals_from_eeg()

    ds = TrainHMSDataset(
        train_df.to_pandas(use_pyarrow_extension_array=True),
        spectrograms=host_specs,
        eeg_specs=spectrograms_from_eeg,
        transform=my_preprocessings.get_transforms(is_train=True),
        signals=signals_from_eeg,
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
    fold: int,
    batch_size: int = 8 * 3,
    num_workers: int = 4,
    is_debug: bool = False,
    remake_specs: bool = False,
    step1: bool = False,
    step2: bool = False,
) -> torch_data.DataLoader[ValidEEGOutput]:
    valid_df = load_valid_df(fold)
    if step1:
        logger.info("Filter less df by votes at 1st stage")
        before_len = len(valid_df)
        valid_df = my_preprocessings.filter_less_df_by_votes(valid_df)
        logger.info(f"Filtered less df by votes at 1st stage: {before_len = } -> {len(valid_df) = }")
    elif step2:
        logger.info("Filter more df by votes at 2nd stage")
        before_len = len(valid_df)
        valid_df = my_preprocessings.filter_more_df_by_votes(valid_df)
        logger.info(f"Filtered more df by votes at 2nd stage: {before_len = } -> {len(valid_df) = }")

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
        spectrograms_from_eeg, signals_from_eeg = _make_specs(valid_df)
    else:
        spectrograms_from_eeg = _load_specs_from_eeg()
        signals_from_eeg = _load_signals_from_eeg()

    ds = ValidHMSDataset(
        valid_df.to_pandas(use_pyarrow_extension_array=True),
        spectrograms=host_specs,
        eeg_specs=spectrograms_from_eeg,
        # transform=my_preprocessings.get_transforms(is_train=True),
        transform=None,
        signals=signals_from_eeg,
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
    import torch.nn as nn
    import torchaudio

    dl = init_train_dataloader(fold=0, is_debug=True)
    batch = next(iter(dl))
    print(batch["x"].shape)
    print(batch["y"].shape)
    print(batch["y"])
    print(f"{batch['y'].min() = }, {batch['y'].max() = }")
    # shape: (bs, 4, 10000)
    print(f"{batch['signals'].shape = }")

    from src.models import layers

    spec_module = layers.init_spec_module()
    dp_transform = layers.init_db_transform_module()
    spec_aug = nn.Sequential(
        # torchaudio.transforms.TimeStretch(fixed_rate=0.6, n_freq=128),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15, iid_masks=False),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=80, iid_masks=False),
        torchaudio.transforms.TimeMasking(time_mask_param=35, iid_masks=False, p=0.5),
        torchaudio.transforms.TimeMasking(time_mask_param=70, iid_masks=False, p=0.5),
    )
    spec_aug = spec_aug.to("cuda")
    # shape: (bs, 4, win_length, hop_length)
    spec = spec_module(batch["signals"])
    spec = dp_transform(spec)
    width = (spec.shape[-1] // 32) * 32
    print(f"1. {spec.shape = }")
    spec = spec[:, :, :, :width]
    print(f"2. {spec.shape = }")
    spec = spec_aug(spec)
    print(f"3. {spec.shape = }")
    spec = spec.permute(0, 2, 3, 1)
    print(f"4. {spec.shape = }")


def _test_train_eeg_ds() -> None:
    from torch.utils import data as torch_data

    from src import constants
    from src.training import preprocessings as my_preprocessings
    from src.utils import common as my_utils_common

    df = load_train_df(fold=0)
    eegs = my_preprocessings.retrieve_eegs_from_parquet(df["eeg_id"].unique().to_list(), constants.feature_cols)
    ds = TrainEEGDataset(df=df.to_pandas(use_pyarrow_extension_array=True), eegs=eegs, transform=None)
    dl: torch_data.DataLoader[TrainEEGOutput] = torch_data.DataLoader(
        dataset=ds,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2,
        worker_init_fn=lambda _: my_utils_common.seed_everything(42),
        persistent_workers=True,
    )

    batch = next(iter(dl))
    print(f"{batch.keys() = }")
    print(f"{batch['x'].shape = }")
    print(f"{batch['y'].shape = }")

    assert batch["x"].shape == (2, 10_000, 20)
    assert batch["y"].shape == (2, 6)


def _test() -> None:
    _test_train_dl()
    # _test_train_eeg_ds()


if __name__ == "__main__":
    _test()
