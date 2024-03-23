import json
import argparse
import dataclasses
import importlib
import pathlib
import pprint

import numpy as np
import pandas as pd
import torch
from torch.utils import data as torch_data

from src import constants, metrics

# from src.exp.exp004 import config as my_config
from src.inference import tools as my_tools
from src.models import common as my_models_common
from src.training import data as my_data
from src.utils import common as my_utils_common
from src.utils import logger as my_utils_logger

logger = my_utils_logger.get_root_logger()


def init_config(exp_ver: str):
    cfg = importlib.import_module(f"src.exp.exp{exp_ver}.config").ConfigImpl()
    return cfg


def _load_dl(exp_ver: str, fold: int, dataloader_params: dict) -> torch_data.DataLoader:
    # dl = importlib.import_module(f"src.exp.exp{exp_ver}.train")._init_valid_dataloader(fold, **dataloader_params)
    # dl = importlib.import_module(f"src.exp.exp{exp_ver}.train").init_valid_dataloader(fold, **dataloader_params)
    dl = my_data.init_valid_dataloader(fold, **dataloader_params)
    return dl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_ver", type=str, default="001")
    args = parser.parse_args()
    exp_ver = args.exp_ver
    # exp_ver = "007"
    # exp_ver = "009"
    # exp_ver = "exp010"
    # cfg = my_config.ConfigImpl()
    cfg = init_config(exp_ver)
    cfg.model_config.model_params.pretrained = False
    # cfg.valid_config.dataloader_params["is_debug"] = True

    logger.info(f"{pprint.pformat(dataclasses.asdict(cfg))}")
    my_utils_common.seed_everything(cfg.seed)
    device = torch.device("cuda")

    print(f"++++++++++++++++++ {exp_ver} +++++++++++++++++++")
    model_weights = [
        # -- Exp001: mean oof score = 0.4823949817313776
        # pathlib.Path("output/exp001/best_exp001_fold0.pth"),
        # pathlib.Path("output/exp001/best_exp001_fold1.pth"),
        # pathlib.Path("output/exp001/best_exp001_fold2.pth"),
        # pathlib.Path("output/exp001/best_exp001_fold3.pth"),
        # pathlib.Path("output/exp001/best_exp001_fold4.pth"),
        # -- Exp002: mean oof score = 0.38010893564018833
        # pathlib.Path("output/exp002/best_exp002_fold0.pth"),
        # pathlib.Path("output/exp002/best_exp002_fold1.pth"),
        # pathlib.Path("output/exp002/best_exp002_fold2.pth"),
        # pathlib.Path("output/exp002/best_exp002_fold3.pth"),
        # pathlib.Path("output/exp002/best_exp002_fold4.pth"),
        # -- Exp003: mean oof score = 0.49379023590927035
        # pathlib.Path("output/exp003/best_exp003_fold0.pth"),
        # pathlib.Path("output/exp003/best_exp003_fold1.pth"),
        # pathlib.Path("output/exp003/best_exp003_fold2.pth"),
        # pathlib.Path("output/exp003/best_exp003_fold3.pth"),
        # pathlib.Path("output/exp003/best_exp003_fold4.pth"),
        # -- Exp004: mean oof score = 0.3813732284941517
        # pathlib.Path("output/exp004/best_exp004_fold0.pth"),
        # pathlib.Path("output/exp004/best_exp004_fold1.pth"),
        # pathlib.Path("output/exp004/best_exp004_fold2.pth"),
        # pathlib.Path("output/exp004/best_exp004_fold3.pth"),
        # pathlib.Path("output/exp004/best_exp004_fold4.pth"),
        # =================================================
        # *[pathlib.Path(f"output/exp{exp_ver}/best_exp{exp_ver}_fold{i}.pth") for i in range(1)],
        *[pathlib.Path(f"output/exp{exp_ver}/best_exp{exp_ver}_fold{i}.pth") for i in range(3)],
        # *[pathlib.Path(f"output/{exp_ver}/best_{exp_ver}_fold{i}.pth") for i in range(5)],
    ]

    scores = []
    for fold in range(5):
        # dl = my_data.init_valid_dataloader(fold, **cfg.valid_config.dataloader_params)
        # dl = my_train._init_valid_dataloader(fold, **cfg.valid_config.dataloader_params)
        dl = _load_dl(exp_ver, fold, cfg.valid_config.dataloader_params)
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
        scores.append({"fold": fold, "score": float(score)})
        logger.info(f"{score = }")

    mean_oof_score = np.mean([score["score"] for score in scores])
    logger.info(f"{mean_oof_score = } { scores = }")
    save_dir = pathlib.Path("output") / f"exp{exp_ver}"
    with save_dir.joinpath("oof_score.json").open("w") as f:
        json.dump({"oof_score": float(mean_oof_score), "scores": scores}, f)


if __name__ == "__main__":
    main()
