import dataclasses
import pathlib
from logging import getLogger
from typing import Literal, Protocol, TypeAlias, TypedDict, TypeVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import models

logger = getLogger(__name__)

ModelType = Literal["HMSModel", "HMSTransformer"]


@dataclasses.dataclass
class HMSModelParams:
    model_name: str
    pretrained: bool


@dataclasses.dataclass
class HMSTransformerModelParams:
    model_name: str
    pretrained: bool


ModelParams: TypeAlias = HMSModelParams | HMSTransformerModelParams


class ModelConfig(Protocol):
    model_name: ModelType
    model_params: ModelParams


def init_model(model_name: ModelType, model_params: ModelParams) -> nn.Module:
    match model_name, model_params:
        case "HMSModel", HMSModelParams(model_name=model_params.model_name, pretrained=model_params.pretrained):
            model = models.HMSModel(model_name=model_params.model_name, pretrained=model_params.pretrained)
            return model
        case "HMSTransformer", HMSTransformerModelParams(
            model_name=model_params.model_name, pretrained=model_params.pretrained
        ):
            model = models.HMSTransformerModel(model_name=model_params.model_name, pretrained=model_params.pretrained)
            return model
        case _:
            raise NotImplementedError(
                f"model_name={model_name} x model_params={model_params} is not supported. Please implement the model"
            )

    # if model_name == "HMSModel":
    #     model = models.HMSModel(model_name=model_params.model_name, pretrained=model_params.pretrained)
    #     return model
    # elif model_name == "HMSTransformer":
    #     model = models.HMSTransformerModel(
    #         model_name=model_params["model_name"], pretrained=model_params["pretrained"]
    #     )
    #     return model

    # else:
    #     raise NotImplementedError(f"model_name={model_name} is not supported. Please implement the model.")


@dataclasses.dataclass
class CompileConfig:
    do_model_compile: bool = True
    compile_mode: Literal["default", "max-autotune"] = "default"
    dynamic_compile: bool = False


_T = TypeVar("_T", bound=torch.nn.Module)


def compile_model(model: _T, compile_config: CompileConfig) -> _T:
    _model = torch.compile(model, mode=compile_config.compile_mode, dynamic=compile_config.dynamic_compile)
    _model = cast(_T, model)
    return _model


def load_model(weights_fp: pathlib.Path, model_name: ModelType, model_params: ModelParams) -> torch.nn.Module:
    logger.info(f"Loading model from {weights_fp}")
    model = init_model(model_name, model_params)
    print(model.load_state_dict(torch.load(weights_fp)))
    return model


class Ensemble(nn.Module):
    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        self.models = models

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        preds = []
        for model in self.models:
            with torch.inference_mode():
                # shape: (bs, 6)
                out = model(x)
                pred = F.softmax(out["logits"], dim=1)
            #                 print(f"{pred = }")
            preds.append(pred.reshape(-1, 1, 6))

        preds = torch.concat(preds, dim=1)
        preds = torch.mean(preds, dim=1).reshape(-1, 6)
        return {"logits": preds}


# ====================
# Test
# ====================
def _test_init_model() -> None:
    model = init_model(
        model_name="HMSModel",
        model_params=HMSModelParams(
            model_name="tf_efficientnet_b0.ns_jft_in1k",
            pretrained=False,
        ),
    )
    print(type(model))

    model = init_model(
        model_name="HMSTransformer",
        model_params=HMSTransformerModelParams(
            model_name="tf_efficientnet_b0.ns_jft_in1k",
            pretrained=False,
        ),
    )
    print(type(model))



def _test_load_model() -> None:
    model = load_model(
        weights_fp=pathlib.Path("output/exp001/last_exp001_fold0.pth"),
        model_name="HMSModel",
        model_params=HMSModelParams(
            model_name="tf_efficientnet_b0.ns_jft_in1k",
            pretrained=False,
        ),
    )
    print(type(model))


def _test() -> None:
    _test_init_model()
    # _test_load_model()


if __name__ == "__main__":
    _test()
