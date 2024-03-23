import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torchaudio
from torch.utils import data as torch_data

from src.models import layers as my_layers
from src.utils import common as my_utils_common


def make_specs_from_signals(
    signals: torch.Tensor,
    spec_module: torchaudio.transforms.Spectrogram,
    time_stretch: torchaudio.transforms.TimeStretch | None = None,
) -> torch.Tensor:
    bs = signals.size(0)
    # signal to specs
    # signal shape: (bs, 4, 4, 10000)
    specs = torch.zeros((bs, 4, 128, 256))
    for k in range(4):
        for kk in range(4):
            # spec shape: (bs, 129, 257)
            spec = spec_module(signals[:, k, kk, :])
            if time_stretch is not None:
                spec = time_stretch(spec)
            spec = spec.abs().pow(2.0)
            # mel_spec = mel_scale_transform(specs)
            # mel_spec = db_transform(mel_spec)

            height = (spec.shape[1] // 32) * 32
            width = (spec.shape[2] // 32) * 32
            spec = spec[:, :height, :width]
            # Standardization -1 to 1
            # mel_spec shape: (bs, 4, 128, 256)

            specs[:, k, :, :] += spec
            specs[:, k, :, :] /= 4

    specs = (specs - specs.mean()) / (specs.std() + 1e-6)
    specs = (specs - specs.min()) / (specs.max() - specs.min() + 1e-6)
    return specs


def inference(
    test_dataloader: torch_data.DataLoader, model: nn.Module, device: torch.device
) -> dict[str, npt.NDArray[np.float32]]:
    model = model.eval().to(device, non_blocking=True)

    n_fft = 1024 // 2 // 2
    hop_length = 10000 // 256
    win_length = 128
    spec_module = my_layers.init_spec_module(power=None, hop_length=hop_length, n_fft=n_fft, win_length=win_length)

    eeg_ids = []
    preds = []
    with torch.inference_mode():
        for _batch_idx, batch in enumerate(test_dataloader):
            my_utils_common.seed_everything(42)
            x = batch["x"].to(device, non_blocking=True, dtype=torch.float32)
            specs = make_specs_from_signals(signals=batch["signals"], spec_module=spec_module)
            specs = specs.permute(0, 2, 3, 1)
            specs = specs.to(device, non_blocking=True, dtype=torch.float32)
            x = torch.cat([x, specs], dim=3)

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
