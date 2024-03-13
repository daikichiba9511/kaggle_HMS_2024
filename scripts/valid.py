import dataclasses
import pathlib
import pprint

import numpy as np
import pandas as pd
import torch

# import torch.nn as nn
# import torch.nn.functional as F
from src import constants, metrics
from src.exp.exp001 import config as my_config
from src.inference import tools as my_tools
from src.models import common as my_models_common
from src.training import data as my_data
from src.utils import common as my_utils_common
from src.utils import logger as my_utils_logger

logger = my_utils_logger.get_root_logger()


def main() -> None:
    cfg = my_config.ConfigImpl()
    cfg.model_config.model_params.pretrained = False
    cfg.valid_config.dataloader_params["is_debug"] = True

    logger.info(f"{pprint.pformat(dataclasses.asdict(cfg))}")
    my_utils_common.seed_everything(cfg.seed)
    device = torch.device("cuda")

    model_weights = [
        # -- Exp001: mean oof score = 0.4823949817313776
        pathlib.Path("output/exp001/best_exp001_fold0.pth"),
        pathlib.Path("output/exp001/best_exp001_fold1.pth"),
        pathlib.Path("output/exp001/best_exp001_fold2.pth"),
        pathlib.Path("output/exp001/best_exp001_fold3.pth"),
        pathlib.Path("output/exp001/best_exp001_fold4.pth"),
    ]

    scores = []
    for fold in range(5):
        dl = my_data.init_valid_dataloader(fold, **cfg.valid_config.dataloader_params)
        models = []
        for model_weight in model_weights:
            if f"{fold}" in model_weight.stem[-1]:
                print(f"Skip fold={fold}, since it's in model_weight.stem {model_weight.stem}")
                continue
            model = my_models_common.load_model(
                weights_fp=model_weight,
                model_name=cfg.model_config.model_name,
                model_params=cfg.model_config.model_params,
            )
            model = model.eval().to(device)
            models.append(model)

        print(f"{len(models) = }")
        ensemble = my_models_common.Ensemble(models=models)
        preds = my_tools.inference(dl, ensemble, device)

        y_preds = preds["predictions"]
        y = np.concatenate([batch["y"].numpy() for batch in dl])
        y_raw = np.concatenate([batch["y_raw"].numpy() for batch in dl])
        print(f"{y_preds.shape = }, {y.shape = }")
        print(f"{y_preds[:3]}")
        print(f"{y[:3]}")

        sub = pd.DataFrame({"eeg_id": preds["eeg_ids"]})
        sub[constants.TARGETS] = y_preds
        print(sub)

        solution = pd.DataFrame({"eeg_id": preds["eeg_ids"]})
        target = y_raw / y_raw.sum(axis=1, keepdims=True)
        solution[constants.TARGETS] = target
        # solution = solution.drop_duplicates(subset=["eeg_id"])
        print(solution)

        score = metrics.score(
            submission=sub,
            solution=solution,
            row_id_column_name="eeg_id",
        )
        scores.append(score)
        logger.info(f"{score = }")

    mean_oof_score = np.mean(scores)
    logger.info(f"{mean_oof_score = } { scores = }")


if __name__ == "__main__":
    main()
