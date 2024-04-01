import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib import axes, figure

pl.Config.set_tbl_cols(100)


def main() -> None:
    print("Analysis of error 39, 40")

    oof_df_path = pathlib.Path("output/exp039/best_oof_fold0.csv")

    oof_df = pl.read_csv(oof_df_path)
    lbl_cols = [c for c in oof_df.columns if "raw" in c and "vote" in c]
    preds_cols = [c for c in oof_df.columns if "predict" in c]
    print(f"oof_df: {oof_df}")
    print(f"lbl_cols: {lbl_cols}, preds_cols: {preds_cols}")

    target = oof_df.to_pandas().loc[:, lbl_cols].to_numpy().argmax(axis=1)
    preds = oof_df.to_pandas().loc[:, preds_cols].to_numpy().argmax(axis=1)
    oof_df = oof_df.with_columns([
        pl.Series("target", target),
        pl.Series("preds", preds),
    ])
    miss_df = oof_df.filter(target != preds).to_pandas()
    print(f"miss_df: {miss_df}")

    print(f"{len(lbl_cols) = }, {len(preds_cols) = }")

    miss_cross_tbl = pd.crosstab(miss_df["target"], miss_df["preds"], normalize="index")
    miss_cross_tbl.loc[:, 3] = 0.0
    miss_cross_tbl.sort_index(axis=0, inplace=True)
    miss_cross_tbl.sort_index(axis=1, inplace=True)
    miss_cross_tbl.index = lbl_cols
    miss_cross_tbl.columns = preds_cols
    print(f"{miss_cross_tbl}")


if __name__ == "__main__":
    main()
