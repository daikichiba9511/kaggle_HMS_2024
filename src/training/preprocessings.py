import pathlib
from collections.abc import Iterable, Sequence
from logging import getLogger

import albumentations as A
import librosa
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from scipy import signal

# import cv2
from tqdm.auto import tqdm

from src import constants

# from albumentations.pytorch import ToTensorV2

logger = getLogger(__name__)

USE_WAVELET = False

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


class Mixup:
    def __init__(self, p: float = 0.5, alpha: float = 0.5) -> None:
        self.p = p
        self.alpha = alpha
        self.lam = 1.0
        self.do_mixup = False

    def init_lambda(self) -> None:
        if np.random.rand() < self.p:
            self.do_mixup = True
        else:
            self.do_mixup = False

        if self.do_mixup and self.alpha > 0.0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1.0

    def reset_lambda(self) -> None:
        self.lam = 1.0


def rand_bbox(width: int, height: int, lam: float) -> tuple[int, int, int, int]:
    """Generate random bounding box.

    Args:
        width: Image width.
        height: Image height.
        lam: Lambda value.
    """
    cut_rate = np.sqrt(1.0 - lam)
    cut_width = int(width * cut_rate)
    cut_height = int(height * cut_rate)

    # uniform sampling
    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_width // 2, 0, width)
    bby1 = np.clip(cy - cut_height // 2, 0, height)
    bbx2 = np.clip(cx + cut_width // 2, 0, width)
    bby2 = np.clip(cy + cut_height // 2, 0, height)
    return bbx1, bby1, bbx2, bby2


def get_transforms(is_train: bool) -> A.Compose:
    if is_train:
        params1 = {
            "num_masks_x": 1,
            "mask_x_length": (0, 20),  # This line changed from fixed  to a range
            "fill_value": (0, 1, 2, 3, 4, 5, 6, 7),
        }
        params2 = {
            "num_masks_y": 1,
            "mask_y_length": (0, 20),
            "fill_value": (0, 1, 2, 3, 4, 5, 6, 7),
        }
        params3 = {
            "num_masks_x": (2, 4),
            "num_masks_y": 5,
            "mask_y_length": 8,
            "mask_x_length": (10, 20),
            "fill_value": (0, 1, 2, 3, 4, 5, 6, 7),
        }
        return A.Compose([
            A.XYMasking(**params1, p=0.5),
            A.XYMasking(**params2, p=0.5),
            A.XYMasking(**params3, p=0.5),
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.OneOf([
                A.XYMasking(mask_x_length=5, mask_y_length=16, num_masks_x=1, num_masks_y=1, fill_value=0, p=0.5),
                A.CoarseDropout(max_holes=4),
            ]),
            # ToTensorV2(),
        ])
    else:
        # return A.Compose([ToTensorV2()])
        raise NotImplementedError("validation transform is not implemented")


def butter_lowpass_filter(
    data: npt.NDArray[np.floating], cutoff_freq: int = 20, samplint_rate: int = 200, order: int = 4
) -> npt.NDArray[np.floating]:
    """Apply a lowpass filter to the data to remove high frequency noise

    Args:
        data: input data
        cutoff_freq: cutoff frequency
        samplint_rate: sampling rate
        order: order of the filter

    Returns:
        filtered_data: filtered data with the same shape as the input data (np.ndarray)
    """
    nyquist = 0.5 * samplint_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = signal.lfilter(b, a, data, axis=0)
    if not isinstance(filtered_data, np.ndarray):
        raise ValueError("filtered_data is not np.ndarray")
    return filtered_data


def mu_law_encoding(data: npt.NDArray[np.floating], mu: int) -> npt.NDArray[np.floating]:
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(1 + mu)
    return mu_x


def quantize_data(data: npt.NDArray[np.floating], classes: int) -> npt.NDArray[np.floating]:
    mu_x = mu_law_encoding(data, classes)
    return mu_x


def preprocess_raw_signal(data: npt.NDArray[np.floating], classes: int = 256) -> npt.NDArray[np.floating]:
    data = butter_lowpass_filter(data)
    data = quantize_data(data, classes)
    return data


def spectrogram_from_egg(parquet_fp: pathlib.Path, offset: int | None = None) -> np.ndarray:
    """Create a spectrogram from the eeg series

    Args:
        parquet_fp: path to the parquet file
        offset: offset to the eeg series

    Returns:
        img: spectrogram

    Notes:
        offset

        |--------------------------------------------------|
        | <----------> | <===========> |
        0    offset
    """
    # Load middle 50 seconds of eeg series
    eeg = pd.read_parquet(parquet_fp)
    if offset is None:
        offset = (len(eeg) - 10_000) // 2

    eeg = eeg.iloc[offset : offset + 10_000]

    img = np.zeros((128, 256, 4), dtype=np.float32)
    signals = []
    for k in range(4):
        cols = FEATS[k]
        for kk in range(4):
            # Compute pair differences
            x = eeg[cols[kk]].to_numpy() - eeg[cols[kk + 1]].to_numpy()

            # TODO: should we use the mean ?
            # Replace nan with the mean
            m = float(np.nanmean(x))
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            signals.append(x)

            # Raw spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=200,
                hop_length=len(x) // 256,
                n_fft=1024,
                n_mels=128,
                fmin=0,
                fmax=20,
                win_length=128,
            )

            # Log transform
            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_db = mel_spec_db[:, :width].astype(np.float32)

            # Standarize -1 to 1
            mel_spec_db = (mel_spec_db + 40) / 40
            img[:, :, k] += mel_spec_db

        # average the 4 montage differences
        img[:, :, k] /= 4
    return img


def load_eeg_from_parquet(fp: pathlib.Path, feature_columns: list[str]) -> npt.NDArray[np.float32]:
    # extract middle 50 seconds
    eeg = pl.read_parquet(fp, columns=feature_columns)
    n_rows = len(eeg)
    offset = (n_rows - 10_000) // 2
    eeg = eeg[offset : offset + 10_000]

    data = np.zeros((10_000, eeg.shape[1]), dtype=np.float32)
    for col_idx, col in enumerate(feature_columns):
        # TODO: should we use the mean ?
        # fill nan with mean
        x = eeg[col].to_numpy().astype(np.float32)
        m = float(np.nanmean(x))
        if np.isnan(x).mean() < 1:
            x = np.nan_to_num(x, nan=m)
        else:
            x[:] = 0

        data[:, col_idx] = x
    return data


def retrieve_eegs_from_parquet(
    eeg_ids: Iterable[str], feature_cols: list[str], dir_path: pathlib.Path = constants.DATA_DIR / "train_eegs"
) -> dict[str, npt.NDArray[np.float32]]:
    retrieved_eegs = {}
    for eeg_id in eeg_ids:
        fp = dir_path / f"{eeg_id}.parquet"
        eeg = load_eeg_from_parquet(fp, feature_columns=feature_cols)
        retrieved_eegs[eeg_id] = eeg
    return retrieved_eegs


def real_kaggle_spectrogram(
    dir_path: pathlib.Path = constants.INPUT_DIR / "brain-spectgrams",
) -> tuple[pl.DataFrame, dict[int, np.ndarray]]:
    """Load the kaggle spectrograms

    Args:
        dir_path: path to the directory where the files are stored

    Returns:
        df_train, specs
    """
    specs = np.load(dir_path / "specs.npy")
    df_train = pl.read_csv(dir_path / "train.csv")
    return df_train, specs


def retrieve_eeg_and_spectrogram(
    df_eeg: pd.DataFrame, df_spectrogram: pd.DataFrame, eeg_offset: int, spectrogram_offset: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve the eeg and spectrogram at the specified offsets

    Args:
        df_eeg:
        df_spectrogram:

    Returns:
        eeg, spec_segment

    References:
    1. https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/468010
    """
    eeg = df_eeg.iloc[eeg_offset * 200 : (eeg_offset + 50) * 200]
    spec_segment = df_spectrogram.query(f"{spectrogram_offset} < time <= {spectrogram_offset + 600}")
    return eeg, spec_segment


def make_spectrograms_from_eeg(
    eeg_ids: Sequence[str],
    dir_path: pathlib.Path = constants.DATA_DIR / "train_eegs",
    with_progress_bar: bool = False,
) -> dict[str, np.ndarray]:
    """Create spectrograms from the eeg files

    Args:
        eeg_ids: list of eeg_ids
        dir_path: path to the directory where the eeg files are stored

    Returns:
        dict: eeg_id -> spectrogram

    """

    if with_progress_bar:
        eeg_ids = tqdm(eeg_ids, total=len(eeg_ids), desc="Create Spectrogram from EEG")  # type: ignore

    spectrograms_from_eeg = {}
    for eeg_id in eeg_ids:
        fp = dir_path / f"{eeg_id}.parquet"
        spectrogram = spectrogram_from_egg(fp, offset=None)
        spectrograms_from_eeg[f"{eeg_id}"] = spectrogram
    return spectrograms_from_eeg


_SPECS_CACHE: dict[str, np.ndarray] | None = None


def load_spectrograms(
    dir_path: pathlib.Path = constants.DATA_DIR / "train_spectrograms",
    with_progress_bar: bool = False,
    is_debug: bool = False,
    fps: Sequence[pathlib.Path] | None = None,
) -> dict[str, np.ndarray]:
    """Load the spectrogram files

    Args:
        dir_path: path to the directory where the spectrogram files are stored

    Returns:
        all_specs: spec_id -> spectrogram

    """
    global _SPECS_CACHE
    if not is_debug and isinstance(_SPECS_CACHE, dict):
        logger.info("Use cache spectrogram")
        return _SPECS_CACHE

    if fps is None:
        fps = list(dir_path.glob("*.parquet"))
        if is_debug:
            fps = fps[:50]

    if with_progress_bar:
        fps = tqdm(fps, total=len(fps), desc="Load Spectrogram")  # type: ignore

    if fps is None:
        raise ValueError("fps is None")

    all_specs = {}
    for fp in fps:
        spec = pd.read_parquet(fp)
        all_specs[fp.stem] = spec.iloc[:, 1:].to_numpy()

    _SPECS_CACHE = all_specs
    return all_specs


def _test_load_spec() -> None:
    specs = load_spectrograms(is_debug=True)
    print(specs)
    print(list(specs.keys())[:5])
    print(specs[list(specs.keys())[0]].shape)


def _test_load_eeg() -> None:
    print(" -- test_load_eeg")
    fp = next(iter(constants.DATA_DIR.joinpath("train_eegs").glob("*.parquet")))
    print(fp)
    eeg = load_eeg_from_parquet(fp, constants.feature_cols)
    print(f"{eeg.shape = }")


def _test() -> None:
    # _test_load_spec()
    _test_load_eeg()


if __name__ == "__main__":
    _test()
