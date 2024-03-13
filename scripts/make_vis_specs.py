import pathlib

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from matplotlib import axes, figure

# from src import constants
from src.training import data as my_data


def plot_specs(specs: npt.NDArray, eegs: npt.NDArray, save_path: pathlib.Path) -> None:
    """plot spectrogram

    Args:
        specs (npt.NDArray): spectrograms to plot
        save_path (pathlib.Path): path to save the plot
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    assert isinstance(fig, figure.Figure)
    assert isinstance(ax, np.ndarray)
    ax[0].imshow(specs, aspect="auto")
    ax[0].set_title("Spectrogram")
    ax[1].imshow(eegs, aspect="auto")
    ax[1].set_title("EEG")

    fig.savefig(save_path)


train_dl = my_data.init_train_dataloader(fold=0, is_debug=True, batch_size=1)
batch = next(iter(train_dl))
print(batch.keys())
print(batch["x"].shape)  # (bs, 8, 128, 256)
print(batch["y"].shape)

spectograms = torch.concat([batch["x"][:, :, :, i : i + 1] for i in range(4)], dim=1)
eeg = torch.concat([batch["x"][:, :, :, i : i + 1] for i in range(4, 8)], dim=1)

print(spectograms.shape)  # (bs, 128, 256, 4)
print(eeg.shape)  # (bs, 128, 256, 4)

plot_specs(spectograms[0].numpy(), eeg[0].numpy(), save_path=pathlib.Path("output/debug.png"))
