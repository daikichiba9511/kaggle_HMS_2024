import pathlib
import random
from collections.abc import Iterable, Sequence
from logging import getLogger
from typing import TypeVar

import albumentations as A
import librosa
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import pywt
import torch
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


def prepare_cutmix(x_max: int, y_max: int, bs: int) -> tuple[list[list[int]], list[int], list[float]]:
    bboxes = []
    k_list = []
    a_list = []
    for _ in range(bs):
        x, y = random.randint(0, x_max), random.randint(0, y_max)
        # width = int(min(x_max, y_max) * random.random())
        cut_width = int(x_max * random.random())
        cut_height = int(y_max * random.random())
        bboxes.append([
            max(0, int(x - cut_width // 2)),
            max(0, int(y - cut_height // 2)),
            min(x_max, int(x + cut_width // 2)),
            min(y_max, int(y + cut_height // 2)),
        ])
        k_list.append(random.randint(0, bs - 1))
        a_list.append(float((cut_width / x_max) * (cut_height / y_max)))
    return bboxes, k_list, a_list


def torch_cutmix(
    img: torch.Tensor, labels: torch.Tensor, a_list: list[float], bboxes: list[list[int]], k_list: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        img: Input image.
            shape (bs, len(kernels), base_filters, round(time_steps/stride))
    """
    # print(f"{img.shape = }")
    max_x, max_y = img.size(3), img.size(2)
    # make cutmix Image
    imgs = []
    lbls = []
    for j in range(img.size(0)):
        # make cutmix image
        k = k_list[j]
        a = a_list[j]
        x1, y1, x2, y2 = bboxes[j]
        one = img[j, :, y1:y2, 0:x1]
        two = img[k, :, y1:y2, x1:x2]
        three = img[j, :, y1:y2, x2:max_x]
        # print(f"{one.shape = }, {two.shape = }, {three.shape = }")
        # shape: (len(kernels), random_length, round(time_steps/stride))
        middle = torch.cat((one, two, three), dim=2)
        # print(f"{img[j, :, 0:y1, :].shape=}, {middle.shape = }, {img[j, :, y2:max_y, :].shape = }")
        imgs.append(torch.cat((img[j, :, 0:y1, :], middle, img[j, :, y2:max_y, :]), dim=1).unsqueeze(0))
        # make cutmix label
        lbl1 = labels[j]
        lbl2 = labels[j]
        lbl = a * lbl1 + (1 - a) * lbl2
        # print(f"{lbl.shape}")
        lbls.append(lbl.reshape(1, -1))
    imgs = torch.concat(imgs, dim=0)
    lbls = torch.concat(lbls, dim=0)
    # print(f"{imgs.shape = }, {lbls.shape = }")
    return imgs, lbls


def cutmix_on_waveform(
    data: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cutmix on waveform

    Args:
        data: input data (bs, 4, 10000)
        labels: labels for the data (bs, 6)
        alpha: alpha value
    """
    bs = data.size(0)
    width = data.size(2)
    sigs = []
    lbls = []
    lams = []
    for i in range(bs):
        lam = np.random.beta(alpha, alpha)
        k = np.random.randint(bs)

        cut_rate = np.sqrt(1.0 - lam)
        cut_width = int(width * cut_rate)
        cx = np.random.randint(width)

        sig = torch.concat([data[i, :, :cx], data[k, :, cx : cx + cut_width], data[i, :, cx + cut_width :]], dim=1)
        lbl = lam * labels[i] + (1 - lam) * labels[k]

        sigs.append(sig.unsqueeze(0))
        lbls.append(lbl.unsqueeze(0))
        lams.append(lam)

    sigs = torch.concat(sigs, dim=0)
    lbls = torch.concat(lbls, dim=0)
    # lams = torch.tensor(lams)
    return sigs, lbls


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
            A.XYMasking(**params1, p=0.75),
            A.XYMasking(**params2, p=0.75),
            A.XYMasking(**params3, p=0.75),
            A.XYMasking(**params3, p=0.75),  # exp028
            # A.HorizontalFlip(p=0.5), # exp001 ~ exp004
            A.VerticalFlip(p=0.5),
            A.OneOf([
                A.XYMasking(mask_x_length=5, mask_y_length=16, num_masks_x=1, num_masks_y=1, fill_value=0, p=0.75),
                A.CoarseDropout(max_holes=4, p=0.75),
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


def denoise_low_freqs(
    data: npt.NDArray[np.floating], wavelet: str = "db4", level: int = 4
) -> npt.NDArray[np.floating]:
    """denoise the low frequency noise in the data using wavelet transform

    Args:
        data: input data, shape (n_channels, n_samples), (4, 10000)

    Returns:
        processed_data: denoised data
    """
    processed_data = []
    for sig in data:
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        # Zero out the approximation coefficients to remove low-frequency noise
        coeffs[0] = np.zeros_like(coeffs[0])
        # Reconstruct the signal using the modified coefficients
        processed_signal = pywt.waverec(coeffs, wavelet)
        processed_data.append(processed_signal)
    processed_data = np.array(processed_data)
    original_signal_length = data.shape[1]
    return processed_data[:, :original_signal_length]


_T = TypeVar("_T", npt.NDArray[np.floating], torch.Tensor)


def mean_std_normalization(data: _T, eps: float = 1e-6) -> _T:
    mean_ = float(data.mean())
    std_ = float(data.std())
    return (data - mean_) / (std_ + eps)


def min_max_normalization(data: _T, eps: float = 1e-6) -> _T:
    min_ = float(data.min())
    max_ = float(data.max())
    return (data - min_) / (max_ - min_ + eps)


def batch_min_max_normalization(data: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    bs = data.size(0)
    data_ = data.view(bs, -1)
    min_ = torch.amin(data_, dim=1)
    max_ = torch.amax(data_, dim=1)
    return (data - min_) / (max_ - min_ + eps)


def preprocess_raw_signal(
    data: npt.NDArray[np.floating], classes: int = 256, reduce_noise: bool = False, channel_normalize: bool = True
) -> npt.NDArray[np.floating]:
    if channel_normalize:
        # normalize data at each channel
        for i in range(data.shape[1]):
            data[i] = min_max_normalization(data[i])
    else:
        data = min_max_normalization(data)

    if reduce_noise:
        data = butter_lowpass_filter(data)
        data = quantize_data(data, classes)
    return data


def add_white_noise(data: torch.Tensor, noise_intensity: float = 0.01) -> torch.Tensor:
    noise = torch.randn_like(data) * noise_intensity
    return data + noise


def augment_signal_takig_label_into_account(
    data: torch.Tensor, labels: torch.Tensor, p: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """

    Args:
        data: input data as signal(=waveform), (bs, n_channels, n_samples)
        labels: labels for the data, (bs, n_classes)

    Returns:
        augmented_data: augmented data
        augmented_labels: augmented labels
    """
    if random.random() > p:
        return data, labels

    # seizure: 0, lpd: 1, gpd: 2, lrda: 3, grda: 4, others: 5
    # 同時に起こりうる (lrda, gpd), (lrda, seizure)
    # 同時に起こらない (gpd, lpd), (grda, lpd)
    # を考慮してaugmentationする
    bs = data.size(0)
    raise NotImplementedError("augment_signal_takig_label_into_account is not implemented")


def spectrogram_from_egg(parquet_fp: pathlib.Path, offset: int | None = None) -> tuple[np.ndarray, np.ndarray]:
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
    # Load middle 50 seconds of eeg serie
    eeg = pd.read_parquet(parquet_fp)
    if offset is None:
        offset = (len(eeg) - 10_000) // 2

    eeg = eeg.iloc[offset : offset + 10_000]

    img = np.zeros((128, 256, 4), dtype=np.float32)
    signals = []
    for k in range(4):
        cols = FEATS[k]
        signal = []
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

            signal.append(x.reshape(1, -1))

            # Raw spectrogram
            # hop_length = len(x) // width
            # n_fft: 横の解像度
            # n_mels: height = n_mesl
            # shape: (n_mels, width) = (128, 256)
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
            # shape: (128, 256)
            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_db = mel_spec_db.astype(np.float32)[:, :width]

            # Standarize -1 to 1
            mel_spec_db = (mel_spec_db + 40) / 40
            img[:, :, k] += mel_spec_db

        # average the 4 montage differences
        img[:, :, k] /= 4
        # shape: (4, 10000)
        signal = np.concatenate(signal, axis=0)
        signals.append(signal.reshape(1, 4, -1))

    # shape: (4, 4, 10000)
    signals = np.concatenate(signals, axis=0)
    return img, signals


def load_eeg_from_parquet(
    fp: pathlib.Path, feature_columns: list[str], sampling_type: str = "normal", sampling_rate: int = 1
) -> npt.NDArray[np.float32]:
    # TODO: extract the middle of the data like 50 seconds or 10 seconds,
    # or extract the middle 50 seconds of the data and then downsampling
    # For example, it extract data at every 4th sample, 50 seconds = 2000 samples
    #
    # extract middle 50 seconds = 10_000 samples
    # extract middle 10 seconds = 2000 samples
    # sequence_length = 10_000
    sequence_length = 2000
    print(f"{sequence_length = }")
    if sampling_type == "normal":
        sample_size = sequence_length // sampling_rate
    else:
        sample_size = sequence_length
    eeg = pl.read_parquet(fp, columns=feature_columns)
    n_rows = len(eeg)
    # mid = n_rows // 2
    # offset = mid - (sample_size // 2)
    #        = (n_rows // 2) - (sample_size // 2)
    #        = (n_rows - sample_size) // 2
    offset = (n_rows - sample_size) // 2
    eeg = eeg[offset : offset + sample_size]
    if sampling_type == "post":
        sample_size = sequence_length // sampling_rate
        eeg = eeg[::sampling_rate]

    data = np.zeros((sequence_length, eeg.shape[1]), dtype=np.float32)
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


def load_eeg_from_parquet2(
    fp: pathlib.Path, feature_columns: list[str], sampling_type: str = "normal", sampling_rate: int = 1
) -> npt.NDArray[np.float32]:
    # TODO: extract the middle of the data like 50 seconds or 10 seconds,
    # or extract the middle 50 seconds of the data and then downsampling
    # For example, it extract data at every 4th sample, 50 seconds = 2000 samples
    #
    # extract middle 50 seconds = 10_000 samples
    # extract middle 10 seconds = 2000 samples

    sequence_length = 10000
    # print(f"{sequence_length = }")
    if sampling_type == "normal":
        sample_size = sequence_length // sampling_rate
    else:
        sample_size = sequence_length
    # eeg = pl.read_parquet(fp, columns=feature_columns)
    eeg = pl.read_parquet(fp)
    n_rows = len(eeg)
    # mid = n_rows // 2
    # offset = mid - (sample_size // 2)
    #        = (n_rows // 2) - (sample_size // 2)
    #        = (n_rows - sample_size) // 2
    offset = (n_rows - sample_size) // 2

    eeg = eeg[offset : offset + sample_size]
    if sampling_type == "post":
        sample_size = 10_000 // sampling_rate
        eeg = eeg[::sampling_rate]

    # data = np.zeros((sample_size, 4), dtype=np.float32)
    signals = []
    for i in range(4):
        cols = FEATS[i]
        signal = []
        for kk in range(4):
            # TODO: should we use the mean ?
            # fill nan with mean
            x = eeg[cols[kk]].to_numpy() - eeg[cols[kk + 1]].to_numpy()
            m = float(np.nanmean(x))
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            signal.append(x.reshape(1, -1))
            # data[:, col_idx] = x
        signal = np.concatenate(signal, axis=0)
        signals.append(signal.reshape(1, 4, -1))
    data = np.concatenate(signals, axis=0)
    return data


def retrieve_eegs_from_parquet(
    eeg_ids: Iterable[str],
    feature_cols: list[str],
    dir_path: pathlib.Path = constants.DATA_DIR / "train_eegs",
    sampling_rate: int = 1,
    sampling_type: str = "normal",
) -> dict[str, npt.NDArray[np.float32]]:
    retrieved_eegs = {}
    for eeg_id in eeg_ids:
        fp = dir_path / f"{eeg_id}.parquet"
        eeg = load_eeg_from_parquet2(
            fp=fp,
            feature_columns=feature_cols,
            sampling_type=sampling_type,
            sampling_rate=sampling_rate,
        )
        eeg = eeg.mean(axis=1).reshape(4, -1).astype(np.float32)
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
) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, npt.NDArray[np.float32]]]:
    """Create spectrograms from the eeg files

    Args:
        eeg_ids: list of eeg_ids
        dir_path: path to the directory where the eeg files are stored

    Returns:
        spectrograms_from_eeg: eeg_id -> spectrogram
        signals_from_eeg: eeg_id -> signals

    """

    if with_progress_bar:
        eeg_ids = tqdm(eeg_ids, total=len(eeg_ids), desc="Create Spectrogram from EEG")  # type: ignore

    spectrograms_from_eeg = {}
    signals_from_eeg = {}
    for eeg_id in eeg_ids:
        fp = dir_path / f"{eeg_id}.parquet"
        spectrogram, signals = spectrogram_from_egg(fp, offset=None)
        spectrograms_from_eeg[f"{eeg_id}"] = spectrogram.astype(np.float32)
        signals_from_eeg[f"{eeg_id}"] = signals.astype(np.float32)
    return spectrograms_from_eeg, signals_from_eeg


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


def filter_more_df_by_votes(df: pl.DataFrame, n_votes: int = 10) -> pl.DataFrame:
    return df.filter(pl.col("total_evaluators") >= n_votes)


def filter_less_df_by_votes(df: pl.DataFrame, n_votes: int = 10) -> pl.DataFrame:
    return df.filter(pl.col("total_evaluators") < n_votes)


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


def _test_load_eeg2() -> None:
    print(" -- test_load_eeg2")
    fp = next(iter(constants.DATA_DIR.joinpath("train_eegs").glob("*.parquet")))
    print(fp)
    eeg = load_eeg_from_parquet2(fp, constants.feature_cols)
    print(f"{eeg.shape = }")
    eeg = eeg.mean(axis=1).reshape(4, -1).astype(np.float32)
    print(f"{eeg.shape = }")


def _test_make_spectrograms() -> None:
    print(" -- test_make_spectrograms")
    fp = next(iter(constants.DATA_DIR.joinpath("train_eegs").glob("*.parquet")))
    print(fp)
    specs, signals = make_spectrograms_from_eeg([fp.stem])
    keys = next(iter(specs.keys()))
    print(f"{keys = }")
    print(f"{specs[keys].shape = }")
    print(f"{signals[keys].shape = }")


def _test() -> None:
    # _test_load_spec()
    # _test_load_eeg()
    _test_load_eeg2()
    # _test_make_spectrograms()


if __name__ == "__main__":
    _test()
