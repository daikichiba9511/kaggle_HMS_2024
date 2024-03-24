import dataclasses
import pathlib
from collections.abc import Sequence
from typing import Any, cast

from src import constants
from src.models import common as my_models_common
from src.training import losses as my_losses
from src.training import optim as my_optim
from src.training import tools as my_tools


@dataclasses.dataclass
class ModelConfigImpl(my_models_common.ModelConfig):
    model_name: my_models_common.ModelType = "HMSTransformer"
    model_params: my_models_common.ModelParams = dataclasses.field(
        default_factory=lambda: my_models_common.HMSTransformerModelParams(
            pretrained=True,
            # model_name="tf_efficientnet_b0.ns_jft_in1k",
            model_name="tf_efficientnet_b2.ns_jft_in1k",
            # model_name="eca_nfnet_l0",
            # model_name="eca_nfnet_l0",
            transformer_model_name="maxvit_small_tf_512",
        )
    )


@dataclasses.dataclass
class TrainConfigImpl(my_tools.TrainConfig):
    n_epochs: int = 15
    optimizer_name: my_optim.OptimizerType = "AdamW"
    optimizer_params: my_optim.OptimParams = dataclasses.field(
        default_factory=lambda: cast(
            my_optim.AdamWParams,
            dict(
                lr=1e-4,
                weight_decay=1e-2,
                eps=1e-6,
            ),
        )
    )
    scheduler_name: my_optim.SchdulerType = "CosineLRScheduler"
    scheduler_params: my_optim.SchduleParams = dataclasses.field(
        default_factory=lambda: cast(
            my_optim.SchduleParams,
            dict(
                t_initial=10,
                lr_min=1e-6,
                warmup_prefix=False,
                warmup_t=1,
                warmup_lr_init=1e-6,
                cycle_limit=10,
            ),
        )
    )
    loss_name: my_losses.LossType = "KLDivLoss"
    metric_names: Sequence[str] = (
        "train/loss",
        "train/duration",
        "valid/loss",
        "valid/duration",
        # -- Task Specific
        # *[f"valid/auc_{i}" for i in range(6)],
        # "valid/acc",
    )
    monitor_metric_name: str = "valid/loss"
    patience: int = 7
    direction: my_tools.EarlyStopDirection = "minimize"
    log_interval: int = 1
    use_amp: bool = True

    dataloader_params: dict[str, Any] = dataclasses.field(
        default_factory=lambda: dict(
            batch_size=8 * 1,  # VRAM 15GB
            # batch_size=8 * 2,
            num_workers=4,
            is_debug=False,
            step1=False,
            step2=True,
        )
    )


@dataclasses.dataclass
class ValidConfigImpl(my_tools.ValidConfig):
    dataloader_params: dict[str, Any] = dataclasses.field(
        default_factory=lambda: dict(
            batch_size=8 * 1,  # VRAM 15GB
            # batch_size=8 * 2,
            num_workers=4,
            is_debug=False,
            step1=False,
            step2=False,
        )
    )


@dataclasses.dataclass
class ConfigImpl(my_tools.Config):
    base: str = "exp030"
    diff: str = "2nd stage training of two stage training w/ all validation data"
    name: str = "exp034"
    seed: int = 42
    output_dir: pathlib.Path = constants.OUTPUT_DIR / name

    model_config: my_models_common.ModelConfig = ModelConfigImpl()
    train_config: my_tools.TrainConfig = TrainConfigImpl()
    valid_config: my_tools.ValidConfig = ValidConfigImpl()


def _test() -> None:
    import pprint

    config = ConfigImpl()
    print(config)
    pprint.pprint(dataclasses.asdict(config))


if __name__ == "__main__":
    _test()
