import pathlib

import matplotlib.pyplot as plt

# import numpy as np
import numpy.typing as npt
import torch
from matplotlib import axes, figure

# from src import constants
from src.training import data as my_data


def plot_specs(specs: npt.NDArray, save_path: pathlib.Path) -> None:
    """plot spectrogram

    Args:
        specs (npt.NDArray): spectrograms to plot
        save_path (pathlib.Path): path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    assert isinstance(fig, figure.Figure)
    assert isinstance(ax, axes.Axes)

    ax.imshow(specs, aspect="auto", origin="lower", cmap="inferno")
    ax.set_title("Spectrogram")

    fig.savefig(save_path)


train_dl = my_data.init_train_dataloader(fold=0, is_debug=True, batch_size=2)
batch = next(iter(train_dl))
print(batch.keys())
print(batch["x"].shape)  # (bs, 8, 128, 256)
print(batch["y"].shape)


spectograms = torch.concat([batch["x"][:, i : i + 1, :, :] for i in range(4)])
eeg = torch.concat([batch["x"][:, i : i + 1, :, :] for i in range(4, 8)])

print(spectograms.shape)  # (bs, 4, 128, 4)
print(eeg.shape)  # (bs, 4, 128, 4)
