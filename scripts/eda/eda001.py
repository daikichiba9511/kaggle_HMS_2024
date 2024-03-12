import pathlib

import polars as pl

from src import constants


from sklearn import model_selection


pl.Config.set_tbl_cols(100)


print(constants.DATA_DIR)
print(list(constants.DATA_DIR.glob("*")))


df_train = pl.read_csv(constants.DATA_DIR / "train.csv")
print(f"{df_train = }")
