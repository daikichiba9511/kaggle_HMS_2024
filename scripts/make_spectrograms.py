import numpy as np
import polars as pl

from src import constants
from src.training import preprocessings as my_preprocessings

train_df = pl.read_csv(constants.DATA_DIR / "train.csv")

print(train_df)

# common preprocessing for all df
train_df = train_df.rename({"spectrogram_id": "spec_id"})
train_eeg_ids = train_df["eeg_id"].unique().to_list()
# eeg_id -> spectrogram
spectrograms_from_eeg, signals_from_eeg = my_preprocessings.make_spectrograms_from_eeg(
    eeg_ids=train_eeg_ids, with_progress_bar=True
)
print(spectrograms_from_eeg)
np.save(constants.OUTPUT_DIR / "spectrograms_from_eeg.npy", spectrograms_from_eeg, allow_pickle=True)  # type: ignore
np.save(constants.OUTPUT_DIR / "signals_from_eeg.npy", signals_from_eeg, allow_pickle=True)  # type: ignore

print("-- Check the saved file --")
# ndarrayにラップされてるので、item()で取り出す
specs = np.load(constants.OUTPUT_DIR / "spectrograms_from_eeg.npy", allow_pickle=True).item()
print(type(specs))
print(specs[list(specs.keys())[0]].shape)
