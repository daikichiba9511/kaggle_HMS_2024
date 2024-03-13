from typing import TypedDict

import timm
import torch
import torch.nn as nn
from typing_extensions import Self

from src.models import feature_extractor as my_feature_extractor


class HMSModel(nn.Module):
    def __init__(self: Self, model_name: str, pretrained: bool) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.backbone.num_features, 6),
        )

    def _consolidate_patches_into_a_image(self: Self, x: torch.Tensor) -> torch.Tensor:
        """make a spectrogram from the eeg and the spectrogram

        Args:
            x (torch.Tensor): input tensor, shape (bs, 128, 256, 8)
        """
        # shape: (bs, 4, 128, 256)
        # spectrograms = [x[:, i : i + 1, :, :] for i in range(4)]
        spectrograms = [x[:, :, :, i : i + 1] for i in range(4)]
        spectrograms = torch.cat(spectrograms, dim=1)

        # shape: (bs, 4, 128, 256)
        eegs = [x[:, :, :, i : i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=1)

        x = torch.cat([spectrograms, eegs], dim=2)
        img = torch.cat([x, x, x], dim=3)

        # region x {spectrogram, eeg} のスペクトログラムの貼り合わせ x 3 (channel direction)
        # |-------------------|
        # |    |    |    |    |
        # |-------------------|
        # |    |    |    |    |
        # |-------------------|
        #
        # permute to (bs, 3, 512, 512) to order channel first
        # shape: (bs, 512, 512, 3) -> (bs, 3, 512, 512)
        img = img.permute(0, 3, 1, 2)
        return img

    def forward(self: Self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        img = self._consolidate_patches_into_a_image(x)
        features = self.features(img)
        logits = self.head(features)

        out = {"logits": logits}
        return out


class HMSTransformerModel(nn.Module):
    def __init__(self: Self, model_name: str, pretrained: bool) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
        )
        self.backbone_transformer = timm.create_model(
            "maxvit_small_tf_512",
            pretrained=pretrained,
            features_only=True,
        )

        # self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1088, 6),
        )

    def _transform_to_a_spectrogram(self: Self, x: torch.Tensor) -> torch.Tensor:
        """make a spectrogram from the eeg and the spectrogram

        Args:
            x (torch.Tensor): input tensor, shape (bs, 128, 256, 8)

        Returns:
            x: transformed tensor, shape (bs, 3, 512, 512)
        """
        # shape: (bs, 4, 128, 256)
        # spectrograms = [x[:, i : i + 1, :, :] for i in range(4)]
        spectrograms = [x[:, :, :, i : i + 1] for i in range(4)]
        spectrograms = torch.cat(spectrograms, dim=1)

        # shape: (bs, 4, 128, 256)
        eegs = [x[:, :, :, i : i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=1)

        x = torch.cat([spectrograms, eegs], dim=2)
        x = torch.cat([x, x, x], dim=3)

        # region x {spectrogram, eeg} のスペクトログラムの貼り合わせ x 3 (channel direction)
        # |-------------------|
        # |    |    |    |    |
        # |-------------------|
        # |    |    |    |    |
        # |-------------------|
        # shape: (bs, 3, 512, 512)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self: Self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._transform_to_a_spectrogram(x)
        x1 = self.backbone_transformer(x)
        # shape: (bs, 768, 16, 16)
        x1 = x1[-1]

        x = self.backbone(x)
        # shape: (bs, 320, 16, 16)
        x = x[-1]

        # 情報の結合の仕方がどれがいいのか, add/concat
        x = torch.cat([x, x1], dim=1)
        logits = self.head(x)

        out = {"logits": logits}
        return out


class FeParams(TypedDict):
    kernels: list[int]
    in_channels: int
    fixed_kernel_size: int


class HMS1DFEModel(nn.Module):
    def __init__(self: Self, fe_params: FeParams) -> None:
        """

        Args:
            fe_params (FeParams):
                kernels (list[int]): list of kernel size to extract features
                in_channels (int): input channel size
                fixed_kernel_size (int): fixed kernel size
            detect_input_feature_size (int): [description]
        """
        super().__init__()
        self.num_classes = 6
        self.fe = my_feature_extractor.Parallel1DConvFeatureExtractor(
            kernels=fe_params["kernels"],
            in_channels=fe_params["in_channels"],
            fixed_kernel_size=fe_params["fixed_kernel_size"],
        )

        self.detector = nn.GRU(
            input_size=fe_params["in_channels"], hidden_size=128, num_layers=1, batch_first=True, bidirectional=True
        )

        self.head = nn.Sequential(
            nn.Linear(in_features=304, out_features=self.num_classes),
        )

    def forward(self: Self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """forward

        Args:
            x (torch.Tensor): input tensor, shape (bs, in_channels, time_steps)

        Returns:
            dict[str, torch.Tensor]:
                logits: output tensor, shape (bs, num_classes)
        """
        hidden = self.fe(x)

        detect_out, _ = self.detector(x.permute(0, 2, 1))
        # 一番最後の状態だけ取り出す
        detect_out = detect_out[:, -1, :]

        out = torch.cat([hidden, detect_out], dim=1)
        logits = self.head(out)
        return {"logits": logits}


# ====================
# Test
# ====================
def _test_hms() -> None:
    print("-- test_hms --")
    model = HMSModel(model_name="tf_efficientnet_b0.ns_jft_in1k", pretrained=False)

    x = torch.rand(2, 128, 256, 8)
    o = model(x)
    print(f"input.shape => {x.shape}")
    print(f"logit.shape => {o['logits'].shape}")
    assert o["logits"].shape == (2, 6), f"Expected shape (2, 6), But got{o['logits'].shape}"


def _test_hms_transformer() -> None:
    print("-- test_hms_transformer --")
    model = HMSTransformerModel(
        model_name="tf_efficientnet_b0.ns_jft_in1k",
        pretrained=False,
    )
    x = torch.rand(2, 128, 256, 8)
    o = model(x)

    print(f"input.shape => {x.shape}")
    print(f"logit.shape => {o['logits'].shape}")
    assert o["logits"].shape == (2, 6), f"Expected shape (2, 6), But got{o['logits'].shape}"


def _test_hms_1dfe_model() -> None:
    print("-- test_hms_1dfe_model --")
    fe_params: FeParams = {
        "kernels": [3, 5, 7, 9],
        "in_channels": 1,
        "fixed_kernel_size": 5,
    }
    model = HMS1DFEModel(fe_params)
    input = torch.rand(2, fe_params["in_channels"], 2500)
    out = model(input)
    print(out["logits"].shape)


def _test() -> None:
    _test_hms()
    _test_hms_transformer()
    _test_hms_1dfe_model()


if __name__ == "__main__":
    _test()
