import timm
import torch
import torch.nn as nn
from typing_extensions import Self


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

    def _transform_to_a_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self.features(x)
        logits = self.head(x)

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


def _test_hms() -> None:
    print("-- test_hms --")
    model = HMSModel(model_name="resnet18", pretrained=False)

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


def _test() -> None:
    _test_hms()
    _test_hms_transformer()


if __name__ == "__main__":
    _test()
