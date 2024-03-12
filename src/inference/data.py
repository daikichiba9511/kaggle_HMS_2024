import pathlib

import albumentations as A
import numpy as np
import pandas as pd
import polars as pl
from torch.utils import data as torch_data

from src import constants
from src.training import preprocessings as my_preprocessings
from src.utils import common as my_utils_common


class TestHMSDataset(torch_data.Dataset):
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
        img = np.ones((128, 256), dtype=np.float32)
        row = self.df.iloc[idx]
        r = 0

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

        if self.transform is not None:
            x = self.transform(image=x)["image"]

        return {
            "x": x,
            "eeg_id": row["eeg_id"],
            "spec_id": row["spec_id"],
        }


def load_test_df(fp: pathlib.Path = constants.DATA_DIR / "test.csv") -> pl.DataFrame:
    test_df = pl.read_csv(fp)

    test_df = test_df.rename({"spectrogram_id": "spec_id"})
    test_df = test_df.with_columns([
        pl.col("spec_id").cast(pl.Utf8),
        pl.col("eeg_id").cast(pl.Utf8),
    ])
    return test_df


def _make_specs(df: pl.DataFrame) -> dict[str, np.ndarray]:
    # spectrograms is made from eeg
    # eeg_id -> spectrogram
    eeg_ids = df["eeg_id"].unique().to_list()
    spectrograms_from_eeg = my_preprocessings.make_spectrograms_from_eeg(
        eeg_ids=eeg_ids, with_progress_bar=True, dir_path=constants.DATA_DIR / "test_eegs"
    )
    print("intersection of spectrograms & eeg_id", len(set(spectrograms_from_eeg.keys()) & set(df["eeg_id"])))
    return spectrograms_from_eeg


def init_test_dataloader(
    fold: int,
    batch_size: int = 8 * 3,
    num_workers: int = 4,
    is_debug: bool = False,
    data_dir: pathlib.Path = constants.DATA_DIR,
) -> torch_data.DataLoader:
    test_df = load_test_df(fp=data_dir / "test.csv")
    # spectrograms was prepared by the host
    # spec_id -> spectrogram
    host_specs = my_preprocessings.load_spectrograms(
        dir_path=data_dir / "test_spectrograms",
        with_progress_bar=True,
        is_debug=is_debug,
    )
    spectrograms_from_eeg = _make_specs(test_df)

    ds = TestHMSDataset(
        test_df.to_pandas(use_pyarrow_extension_array=True),
        spectrograms=host_specs,
        eeg_specs=spectrograms_from_eeg,
        transform=my_preprocessings.get_transforms(is_train=True),
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
