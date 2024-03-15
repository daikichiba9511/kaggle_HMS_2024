import pathlib

import matplotlib.pyplot as plt
import polars as pl
from torch.utils import data as torch_data

from src import constants
from src.training import data as my_data
from src.training import preprocessings as my_preprocessings
from src.utils import common as my_utils_common

plt.style.use("ggplot")

SAVE_DIR = pathlib.Path("output/visualizations/signals_of_train_eeg_dataset")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

df = my_data.load_train_df(fold=0)
eeg_ids = df["eeg_id"].unique().to_list()[:10]
eegs = my_preprocessings.retrieve_eegs_from_parquet(eeg_ids, constants.feature_cols)
df = df.filter(pl.col("eeg_id").is_in(eeg_ids))
ds = my_data.TrainEEGDataset(df=df.to_pandas(use_pyarrow_extension_array=True), eegs=eegs, transform=None)
dl = torch_data.DataLoader(
    dataset=ds,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    drop_last=True,
    worker_init_fn=lambda _: my_utils_common.seed_everything(42),
)

for batch_idx, batch in enumerate(dl):
    # shape: (batch_size, n_channels, n_times)
    x = batch["x"]
    # shape: (batch_size, n_classes)
    y = batch["y"]

    print(f"{x.shape = }, {y.shape = }")

    eeg_ids = batch["eeg_id"]
    spec_ids = batch["spec_id"]

    fig, axs = plt.subplots(5, 4, figsize=(20, 10))
    fig.suptitle(f"Fold={0}, eeg_id={eeg_ids[0]}, spec_id={spec_ids[0]}, \n {y.tolist()[0]}")
    for i in range(5):
        for j in range(4):
            idx = i * 4 + j
            signal = x[0, idx, :]
            target = y[0]
            # print(f"{signal.shape = }, {target.shape = }")
            print(f"## {idx} signal value range: [{signal.min()}, {signal.max()}]")
            axs[i, j].plot(signal)
            axs[i, j].set_title(f"ch{idx}")
            axs[i, j].set_ylim(-3, 3)
            if i == 4:
                axs[i, j].set_xlabel("time")
            if j == 0:
                axs[i, j].set_ylabel("standarized value")

    fig.tight_layout()
    fig.savefig(SAVE_DIR / f"fold0_eeg_id{eeg_ids[0]}_spec_id{spec_ids[0]}.png")

    if batch_idx > 10:
        break
