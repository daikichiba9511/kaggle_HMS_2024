import pandas as pd

from src import constants

df = pd.read_csv(constants.DATA_DIR / "train.csv")
print(df)

train_df = df.groupby("eeg_id")[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg({
    "spectrogram_id": "first",
    "spectrogram_label_offset_seconds": "min",
})
train_df.columns = ["spectrogram_id", "min"]
print(train_df)

aux = df.groupby("eeg_id")[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg({
    "spectrogram_label_offset_seconds": "max",
})
print(aux)
train_df.loc[:, "max"] = aux
print(train_df)

aux = df.groupby("eeg_id")[["patient_id"]].agg("first")
train_df.loc[:, "patient_id"] = aux

aux = df.groupby("eeg_id")[constants.TARGETS].agg("sum")
for label in constants.TARGETS:
    train_df.loc[:, label] = aux[label]

print(train_df)

aux = df.groupby("eeg_id")["expert_consensus"].agg("first")
train_df.loc[:, "target"] = aux

train_df = train_df.reset_index()
print(f"{train_df = }")
print(f"{len(train_df) / len(df) = }, ({len(train_df) = } / {len(df)})")
print(f"{train_df.columns = }")
train_df.to_csv(constants.DATA_DIR / "train_preprocessed.csv", index=False)
