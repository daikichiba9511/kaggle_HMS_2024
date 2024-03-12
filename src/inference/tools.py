import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils import data as torch_data

from src.utils import common as my_utils_common


def inference(
    test_dataloader: torch_data.DataLoader, model: nn.Module, device: torch.device
) -> dict[str, npt.NDArray[np.float32]]:
    model = model.eval().to(device, non_blocking=True)

    eeg_ids = []
    preds = []
    with torch.inference_mode():
        for _batch_idx, batch in enumerate(test_dataloader):
            my_utils_common.seed_everything(42)
            x = batch["x"].to(device, non_blocking=True, dtype=torch.float32)
            out = model(x)
            logits = out["logits"]
            y_preds = logits.detach().cpu().numpy()
            eeg_ids.append(batch["eeg_id"])
            preds.append(y_preds)

    preds_dict = {
        "predictions": np.concatenate(preds).astype(np.float32),
        "eeg_ids": np.concatenate(eeg_ids).astype(str),
    }
    return preds_dict


# ========================
# -- Test of Inference --
# ========================
def _test_inference() -> None:
    from typing import cast

    import pandas as pd

    from src import constants
    from src.inference.data import init_test_dataloader
    from src.models import common as my_models_common

    preds = inference(
        init_test_dataloader(0),
        my_models_common.init_model(
            model_name="HMSModel",
            model_params=cast(
                my_models_common.HMSModelParams,
                dict(
                    pretrained=False,
                    model_name="tf_efficientnet_b0.ns_jft_in1k",
                ),
            ),
        ),
        torch.device("cpu"),
    )
    print(preds)
    print(preds["predictions"].shape)

    sub = pd.DataFrame({"eeg_id": preds["eeg_ids"]})
    sub[constants.TARGETS] = preds["predictions"]
    print(sub)


def _test() -> None:
    _test_inference()


if __name__ == "__main__":
    _test()
