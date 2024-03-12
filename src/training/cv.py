from logging import getLogger

import polars as pl
from sklearn import model_selection

logger = getLogger(__name__)


def make_fold(
    df: pl.DataFrame, group_col_name: str, target_col_name: str = "target", n_splits: int = 5
) -> pl.DataFrame:
    kf = model_selection.GroupKFold(n_splits=n_splits)
    _df = df.with_columns(pl.Series("fold", [0] * df.height)).to_pandas(use_pyarrow_extension_array=True)
    for fold, (_, valid_idx) in enumerate(kf.split(X=df, y=df[target_col_name], groups=df[group_col_name])):
        _df.loc[valid_idx, "fold"] = fold
    df = pl.from_pandas(_df)
    return df


def _test() -> None:
    import matplotlib.pyplot as plt

    # for plotting
    import numpy as np
    from matplotlib import axes, figure

    from src import constants

    df = pl.read_csv(constants.DATA_DIR / "train.csv")
    print(df)
    df = make_fold(df, group_col_name="patient_id", target_col_name="expert_consensus")
    print(df)
    print(df.columns)
    print(df["fold"].value_counts())

    # colの値は、特定の活動だと思うアノテーターの数なので1以上で1それ以外な0に変換する
    df_for_plot = df.select(["fold", "seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"])
    print(df_for_plot)
    df_for_plot = df_for_plot.with_columns([
        pl.when(pl.col(col) > 1).then(1).otherwise(0).alias(col.split("_")[0])
        for col in df_for_plot.columns
        if col != "fold"
    ])
    print(df_for_plot)

    save_path = constants.OUTPUT_DIR / "eda" / "eda001"
    save_path.mkdir(parents=True, exist_ok=True)

    labels = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]

    # 各folgごとにラベルの分布を出す
    for fold in range(df["fold"].n_unique()):
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        assert isinstance(ax, axes.Axes), f"{type(axes)=}"
        assert isinstance(fig, figure.Figure)

        x = np.arange(6)
        y = df_for_plot.filter(pl.col("fold") == fold).select(labels).to_pandas().sum()

        ax.bar(x=x, height=y, tick_label=labels)
        ax.set_title(f"fold={fold}")
        ax.set_xlabel("vote")
        ax.set_ylabel("count")

        ax.set_ylim(0, 8000)

        ax.grid(False)
        fig.tight_layout()
        fig.savefig(save_path / f"fold_{fold}.png")


if __name__ == "__main__":
    _test()
