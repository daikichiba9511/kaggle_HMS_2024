import dataclasses

import timm
import torch
import torch.nn as nn
from typing_extensions import Self

from src.models import feature_extractor as my_feature_extractor
from src.models import layers as my_layers
from src.training import preprocessings as my_preprocess


@dataclasses.dataclass
class HMSModelParams:
    model_name: str
    pretrained: bool


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

        Returns:
            x: transformed tensor, shape (bs, 3, 512, 512)
        """
        # spectrograms shape: (bs, 128, 256, 4)
        spectrograms = torch.cat([x[:, :, :, i : i + 1] for i in range(4)], dim=1)
        # eegs shape: (bs, 512, 256, 4)
        eegs = torch.cat([x[:, :, :, i : i + 1] for i in range(4, 8)], dim=1)
        x1 = torch.cat([spectrograms, eegs], dim=2)

        # specs_from_signals shape: (bs, 512, 256, 4)
        specs_from_signals = torch.cat([x[:, :, :, i : i + 1] for i in range(8, 12)], dim=1)
        x2 = torch.cat([specs_from_signals, specs_from_signals], dim=2)

        # x shape: (bs, 512, 512, 3)
        x = torch.cat([x1, x1, x1], dim=3)
        # x = torch.cat([x1, x2, x1], dim=3)

        # region x {spectrogram, eeg} のスペクトログラム(128, 256)の貼り合わせ x 3 (channel direction)
        # |-----------|
        # |  |  |  |  |
        # |  |  |  |  |
        # |-----------|
        # |  |  |  |  |
        # |  |  |  |  |
        # |-----------|
        # shape: (bs, 3, 512, 512)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(
        self: Self,
        x: torch.Tensor,
        do_mixup: bool = False,
        lam: float | None = None,
        y: torch.Tensor | None = None,
        bboxes: list[list[int]] | None = None,
        k_list: list[int] | None = None,
        a_list: list[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        # img shape: (bs, 3, 512, 512)
        img = self._consolidate_patches_into_a_image(x)
        # -- CutMixup
        if lam is not None and do_mixup:
            if bboxes is not None and y is not None and k_list is not None and a_list is not None:
                img, y = my_preprocess.torch_cutmix(
                    img=img,
                    labels=y,
                    a_list=a_list,
                    bboxes=bboxes,
                    k_list=k_list,
                )
                # Mixup
                img = lam * img + (1.0 - lam) * img.flip(0)
                y = lam * y + (1.0 - lam) * y.flip(0)
            else:
                img = lam * img + (1.0 - lam) * img.flip(0)
                y = None

        features = self.features(img)
        logits = self.head(features)

        out = {"logits": logits}
        if y is not None:
            out["y"] = y
        return out


@dataclasses.dataclass
class HMSTransformerModelParams:
    model_name: str
    pretrained: bool
    transformer_model_name: str = "maxvit_small_tf_512"


class HMSTransformerModel(nn.Module):
    def __init__(
        self: Self, model_name: str, pretrained: bool, transformer_model_name: str = "maxvit_small_tf_512"
    ) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
        )
        self.backbone_transformer = timm.create_model(
            transformer_model_name,
            pretrained=pretrained,
            features_only=True,
        )

        # self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        backbone_num_features = self.backbone.feature_info.channels()[-1]
        backbone_transformer_num_features = self.backbone_transformer.feature_info[-1]["num_chs"]
        self.num_features = backbone_num_features + backbone_transformer_num_features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(self.num_features, 6),
        )

    def _consolidate_patches_into_a_image(self: Self, x: torch.Tensor) -> torch.Tensor:
        """make a spectrogram from the eeg and the spectrogram

        Args:
            x (torch.Tensor): input tensor, shape (bs, 128, 256, 8)

        Returns:
            x: transformed tensor, shape (bs, 3, 512, 512)
        """
        # shape: (bs, 128, 256, 4)
        # spectrograms = [x[:, i : i + 1, :, :] for i in range(4)]
        spectrograms = [x[:, :, :, i : i + 1] for i in range(4)]
        # shape: (bs, 512, 256, 4)
        spectrograms = torch.cat(spectrograms, dim=1)

        # shape: (bs, 128, 256, 4)
        eegs = [x[:, :, :, i : i + 1] for i in range(4, 8)]
        # shape: (bs, 512, 256, 4)
        eegs = torch.cat(eegs, dim=1)

        if x.shape[3] == 12:
            # shape: (bs, 128, 256, 4)
            specs_from_signals = [x[:, :, :, i : i + 1] for i in range(8, 12)]
            # shape: (bs, 512, 256, 4)
            specs_from_signals = torch.cat(specs_from_signals, dim=1)

            # specs_from_signals = torch.cat([specs_from_signals, specs_from_signals], dim=2)

            # x = torch.cat([spectrograms, eegs], dim=2)
            x = torch.cat([specs_from_signals, eegs], dim=2)
            # x2 = torch.cat([eegs, specs_from_signals], dim=2)
            x = torch.cat([x, x, x], dim=3)
            # exp026
            # x = torch.cat([x, x2, x], dim=3)
            # x = torch.cat([x, x2, x], dim=3)
            # exp027
            # x = torch.cat([x, specs_from_signals, specs_from_signals], dim=3)
        else:
            x = torch.cat([spectrograms, eegs], dim=2)
            x = torch.cat([x, x, x], dim=3)

        # region x {spectrogram, eeg} のスペクトログラム(128, 256)の貼り合わせ x 3 (channel direction)
        # |-----------|
        # |  |  |  |  |
        # |  |  |  |  |
        # |-----------|
        # |  |  |  |  |
        # |  |  |  |  |
        # |-----------|
        # shape: (bs, 3, 512, 512)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(
        self: Self,
        x: torch.Tensor,
        do_mixup: bool = False,
        lam: float | None = None,
        y: torch.Tensor | None = None,
        bboxes: list[list[int]] | None = None,
        k_list: list[int] | None = None,
        a_list: list[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        img = self._consolidate_patches_into_a_image(x)
        # -- CutMixup
        if lam is not None and do_mixup:
            if bboxes is not None and y is not None and k_list is not None and a_list is not None:
                x, y = my_preprocess.torch_cutmix(
                    img=img,
                    labels=y,
                    a_list=a_list,
                    bboxes=bboxes,
                    k_list=k_list,
                )
                img = lam * x + (1.0 - lam) * x.flip(0)
                y = lam * y + (1.0 - lam) * y.flip(0)
            else:
                img = lam * img + (1.0 - lam) * img.flip(0)
                y = None

        x1 = self.backbone_transformer(img)
        # shape: (bs, 768, 16, 16)
        x1 = x1[-1]

        x = self.backbone(img)
        # shape: (bs, 320, 16, 16)
        x = x[-1]

        # 情報の結合の仕方がどれがいいのか, add/concat
        x = torch.cat([x, x1], dim=1)
        logits = self.head(x)

        out = {"logits": logits}
        if y is not None:
            out["y"] = y
        return out


@dataclasses.dataclass
class HMS1DSpecTransformerModelParams:
    model_name: str
    pretrained: bool
    transformer_model_name: str = "maxvit_ti_tf_512"


class HMS1DSpecTransformerModel(nn.Module):
    def __init__(
        self: Self, model_name: str, pretrained: bool, transformer_model_name: str = "maxvit_tiny_tf_512"
    ) -> None:
        super().__init__()
        self.pretrained = pretrained
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
        )
        self.backbone_transformer = timm.create_model(
            transformer_model_name,
            pretrained=pretrained,
            features_only=True,
        )

        backbone_num_features = self.backbone.feature_info.channels()[-1]
        backbone_transformer_num_features = self.backbone_transformer.feature_info[-1]["num_chs"]
        self.num_features = backbone_num_features + backbone_transformer_num_features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(self.num_features, 6),
        )
        self.cnn_spec = my_feature_extractor.CNNSpectgram(
            in_channels=4,
            base_filters=128,  # height of the image
            kernel_size=(128, 64, 32, 16),  # out_channels = len(kernel_size)
            stride=10000 // 256,  # stride = len(x) // width
            reinit=True,
        )

    def _consolidate_patches_into_a_image(self: Self, x: torch.Tensor) -> torch.Tensor:
        """make a spectrogram from the eeg and the spectrogram

        Args:
            x (torch.Tensor): input tensor, shape (bs, 128, 256, 8/12/16)
                8: spectrogram + eeg
                12: spectrogram + eeg + spectrogram from signals
                16: spectrogram + eeg + spectrogram from signals + wavegram

        Returns:
            x: transformed tensor, shape (bs, 3, 512, 512)
        """
        # shape: (bs, 128, 256, 4)
        # spectrograms = [x[:, i : i + 1, :, :] for i in range(4)]
        spectrograms = [x[:, :, :, i : i + 1] for i in range(4)]
        # shape: (bs, 512, 256, 4)
        spectrograms = torch.cat(spectrograms, dim=1)

        # shape: (bs, 128, 256, 4)
        eegs = [x[:, :, :, i : i + 1] for i in range(4, 8)]
        # shape: (bs, 512, 256, 4)
        eegs = torch.cat(eegs, dim=1)

        if x.shape[3] == 12:
            # shape: (bs, 128, 256, 4)
            specs_from_signals = [x[:, :, :, i : i + 1] for i in range(8, 12)]
            # shape: (bs, 512, 256, 4)
            specs_from_signals = torch.cat(specs_from_signals, dim=1)
            specs_from_signals = torch.cat([specs_from_signals, specs_from_signals], dim=2)

            x = torch.concat([spectrograms, eegs], dim=2)
            # exp026
            x = torch.concat([x, specs_from_signals, x], dim=3)
            # exp027
            # x = torch.cat([x, specs_from_signals, specs_from_signals], dim=3)
        elif x.shape[3] == 16:
            x = torch.concat([spectrograms, eegs], dim=2)

            # spec_from_signals shape: (bs, 128, 256, 4)
            specs_from_signals = torch.concat([x[:, :, :, i : i + 1] for i in range(8, 12)], dim=1)
            x2 = torch.concat([specs_from_signals, specs_from_signals], dim=2)

            # wavegram shape: (bs, 128, 256, 4)
            wavegram = torch.cat([x[:, :, :, i : i + 1] for i in range(12, 16)], dim=1)
            x3 = torch.concat([wavegram, wavegram], dim=2)

            x = torch.concat([x, x2, x3], dim=3)

        else:
            x = torch.concat([spectrograms, eegs], dim=2)
            x = torch.concat([x, x, x], dim=3)

        # region x {spectrogram, eeg} のスペクトログラム(128, 256)の貼り合わせ x 3 (channel direction)
        # |-----------|
        # |  |  |  |  |
        # |  |  |  |  |
        # |-----------|
        # |  |  |  |  |
        # |  |  |  |  |
        # |-----------|
        # shape: (bs, 3, 512, 512)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(
        self: Self,
        x: torch.Tensor,
        do_mixup: bool = False,
        lam: float | None = None,
        y: torch.Tensor | None = None,
        bboxes: list[list[int]] | None = None,
        k_list: list[int] | None = None,
        a_list: list[float] | None = None,
        signals: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if signals is not None:
            # shape: (bs, 4, 128, 256)
            wavegram = self.cnn_spec(signals)
            x = torch.cat([x, wavegram.permute(0, 2, 3, 1)], dim=-1)

        img = self._consolidate_patches_into_a_image(x)
        # -- CutMixup
        if lam is not None and do_mixup:
            if bboxes is not None and y is not None and k_list is not None and a_list is not None:
                x, y = my_preprocess.torch_cutmix(
                    img=img,
                    labels=y,
                    a_list=a_list,
                    bboxes=bboxes,
                    k_list=k_list,
                )
                img = lam * x + (1.0 - lam) * x.flip(0)
                y = lam * y + (1.0 - lam) * y.flip(0)
            else:
                img = lam * img + (1.0 - lam) * img.flip(0)
                y = None

        x1 = self.backbone_transformer(img)
        # shape: (bs, 768, 16, 16)
        x1 = x1[-1]

        x = self.backbone(img)
        # shape: (bs, 320, 16, 16)
        x = x[-1]

        # 情報の結合の仕方がどれがいいのか, add/concat
        x = torch.cat([x, x1], dim=1)
        x = self.pool(x)
        logits = self.head(x)

        out = {"logits": logits}
        if y is not None:
            out["y"] = y
        return out


@dataclasses.dataclass
class GRUParams:
    hidden_size: int = 128
    num_layers: int = 1


@dataclasses.dataclass
class HMS1DParallelConvParams:
    kernels: list[int]
    in_channels: int
    fixed_kernel_size: int

    gru_params: GRUParams | None = GRUParams()


class HMS1DParallelConvModel(nn.Module):
    def __init__(self: Self, fe_params: HMS1DParallelConvParams) -> None:
        """

        Args:
            fe_params:

                * kernels (list[int]): list of kernel size to extract features
                * in_channels (int): input channel size
                * fixed_kernel_size (int): fixed kernel size

        """
        super().__init__()
        self.num_classes = 6
        self.fe = my_feature_extractor.Parallel1DConvFeatureExtractor(
            kernels=fe_params.kernels,
            in_channels=fe_params.in_channels,
            fixed_kernel_size=fe_params.fixed_kernel_size,
        )
        if fe_params.gru_params is not None:
            params = fe_params.gru_params
        else:
            raise ValueError("gru_params is not set")

        self.detector = nn.GRU(
            input_size=1,
            hidden_size=params.hidden_size,
            num_layers=params.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.pooling_out_size = 256
        self.pooling = nn.AdaptiveAvgPool1d(self.pooling_out_size)

        head_size = 2 * params.hidden_size * self.pooling_out_size
        self.head = nn.Sequential(
            nn.Linear(in_features=head_size, out_features=self.num_classes),
        )

    def forward(self: Self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """forward

        Args:
            x (torch.Tensor): input tensor, shape (bs, in_channels, time_steps)

        Returns:
            dict[str, torch.Tensor]:
                logits: output tensor, shape (bs, num_classes)
        """
        bs = x.shape[0]
        # shape: (bs, hidden_size1)
        hidden = self.fe(x)

        # detect_out shape: (bs, sequence_length, 2 * hidden_size)
        detect_out, _ = self.detector(hidden.unsqueeze(-1))
        # detect out shape: (bs, 2 * hidden_size, pooling_out_size)
        detect_out = self.pooling(detect_out.permute(0, 2, 1))

        out = detect_out.reshape(bs, -1)
        logits = self.head(out)
        return {"logits": logits}


@dataclasses.dataclass
class HMS1DParallel2WayConvParams:
    short_kernels: list[int]
    long_kernels: list[int]
    in_channels: int
    fixed_kernel_size: int

    gru_params: GRUParams | None = GRUParams()


class HMS1DParallel2WayConvModel(nn.Module):
    def __init__(self: Self, fe_params: HMS1DParallel2WayConvParams) -> None:
        """

        Args:
            fe_params:

                * short_kernels (list[int]): list of kernel size to extract features
                * long_kernels (list[int]): list of kernel size to extract features
                * in_channels (int): input channel size
                * fixed_kernel_size (int): fixed kernel size

        """
        super().__init__()
        self.num_classes = 6
        self.short_fe = my_feature_extractor.Parallel1DConvFeatureExtractor(
            kernels=fe_params.short_kernels,
            in_channels=fe_params.in_channels,
            fixed_kernel_size=fe_params.fixed_kernel_size,
        )
        self.long_fe = my_feature_extractor.Parallel1DConvFeatureExtractor(
            kernels=fe_params.long_kernels,
            in_channels=fe_params.in_channels,
            fixed_kernel_size=fe_params.fixed_kernel_size,
        )
        if fe_params.gru_params is not None:
            params = fe_params.gru_params
        else:
            raise ValueError("gru_params is not set")

        self.detector = nn.GRU(
            # input_size=fe_params.in_channels,
            input_size=1,
            hidden_size=params.hidden_size,
            num_layers=params.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.pooling_out_size = 256
        self.pooling = nn.AdaptiveAvgPool1d(self.pooling_out_size)

        head_size = 2 * params.hidden_size * self.pooling_out_size
        self.head = nn.Sequential(
            # nn.LayerNorm(head_size),
            nn.Dropout(0.5, inplace=True),
            nn.Linear(in_features=head_size, out_features=self.num_classes),
        )

    def forward(self: Self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """forward

        Args:
            x (torch.Tensor): input tensor, shape (bs, in_channels, time_steps)

        Returns:
            dict[str, torch.Tensor]:
                logits: output tensor, shape (bs, num_classes)
        """
        # shape: (bs, hidden_size1)
        short_hidden = self.short_fe(x)
        long_hidden = self.long_fe(x)

        # shape: (bs, hidden_size1 + hidden_size2, 1)
        hidden = torch.cat([short_hidden, long_hidden], dim=1).unsqueeze(-1)
        detect_out, _ = self.detector(hidden)
        out = self.pooling(detect_out.permute(0, 2, 1)).reshape(x.shape[0], -1)

        # out = torch.cat([short_hidden, long_hidden, out], dim=1)
        logits = self.head(out)
        return {"logits": logits}


@dataclasses.dataclass
class HMSCNNSpecFEParams:
    in_channels: int = 4
    base_filters: int = 128
    kernel_size: tuple[int, ...] = (32, 16, 4, 2)
    stride: int = 4
    sigmoid: bool = False
    output_size: int | None = None
    reinit: bool = True
    encoder_name: str = "tf_efficientnet_b0_ns"
    encoder_pretrained: bool = True


class HMSCNNSpecFEModel(nn.Module):
    def __init__(self: Self, params: HMSCNNSpecFEParams) -> None:
        super().__init__()
        self.params = params
        self.num_classes = 6
        self.fe = my_feature_extractor.CNNSpectgram(
            in_channels=params.in_channels,
            base_filters=params.base_filters,
            kernel_size=params.kernel_size,
            stride=params.stride,
            sigmoid=params.sigmoid,
            output_size=params.output_size,
            reinit=params.reinit,
        )
        self.encoder = timm.create_model(
            params.encoder_name,
            params.encoder_pretrained,
            features_only=True,
            in_chans=len(params.kernel_size),
            # in_chans=3,
            drop_rate=0.5,
        )
        self.pool = my_layers.GeM()
        self.num_features = self.encoder.feature_info.channels()[-1]
        self.head = nn.Sequential(
            # nn.LayerNorm(self.num_features),
            nn.Linear(in_features=self.num_features, out_features=self.num_classes),
            nn.Dropout(0.5, inplace=True),
        )

    def forward(
        self: Self,
        x: torch.Tensor,
        do_mixup: bool = False,
        lam: float | None = None,
        y: torch.Tensor | None = None,
        bboxes: list[list[int]] | None = None,
        k_list: list[int] | None = None,
        a_list: list[float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: input tensor, shape (bs, time_steps, in_channels)
        """

        # shape: (bs, len(kernels), base_filters, time_steps//stride)
        spec_img = self.fe(x)
        # spec_img = [spec_img[:, i, :, :] for i in range(spec_img.shape[1])]
        # spec_img = torch.cat(spec_img, dim=1).unsqueeze(1)
        # spec_img = torch.cat([spec_img, spec_img, spec_img], dim=1)
        # w = spec_img.shape[2]
        # spec_img = spec_img[:, :, :, :w]

        # -- CutMixup
        if lam is not None and do_mixup:
            if bboxes is not None and y is not None and k_list is not None and a_list is not None:
                x, y = my_preprocess.torch_cutmix(
                    img=spec_img,
                    labels=y,
                    a_list=a_list,
                    bboxes=bboxes,
                    k_list=k_list,
                )
                x = lam * x + (1.0 - lam) * x.flip(0)
                y = lam * y + (1.0 - lam) * y.flip(0)
            else:
                x = lam * spec_img + (1.0 - lam) * spec_img.flip(0)
                y = None

        features = self.encoder(spec_img)
        last_layer_feature = features[-1]

        pooled_features = self.pool(last_layer_feature)
        pooled_features = pooled_features.view(x.shape[0], -1)

        logits = self.head(pooled_features)

        out = {"logits": logits}
        if lam is not None and do_mixup and y is not None:
            out["y"] = y
        return out


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
    # x = torch.rand(2, 128, 256, 8)
    x = torch.rand(2, 128, 256, 12)
    o = model(x)

    print(f"input.shape => {x.shape}")
    print(f"logit.shape => {o['logits'].shape}")
    assert o["logits"].shape == (2, 6), f"Expected shape (2, 6), But got{o['logits'].shape}"


def _test_hms_1dfe_model() -> None:
    print("-- test_hms_1dfe_model --")
    fe_params = HMS1DParallelConvParams(
        kernels=[3, 5, 7, 9],
        in_channels=20,
        fixed_kernel_size=5,
    )

    model = HMS1DParallelConvModel(fe_params)
    input = torch.rand(2, fe_params.in_channels, 10000)
    out = model(input)
    print(out["logits"].shape)


def _test_hms_1d_parallel_2way_conv() -> None:
    print("-- test_hms_1d_paralell_2way_conv --")
    fe_params = HMS1DParallel2WayConvParams(
        short_kernels=[3, 5, 7, 9],
        long_kernels=[128, 64, 32, 16],
        in_channels=20,
        fixed_kernel_size=5,
    )

    model = HMS1DParallel2WayConvModel(fe_params)
    input = torch.rand(2, fe_params.in_channels, 10000)
    out = model(input)
    print(out["logits"].shape)


def _test_hms_cnn_spec_fe() -> None:
    import random

    from src.training import preprocessings as my_preprocess

    print("-- test_hms_cnn_spec_fe --")
    fe_params = HMSCNNSpecFEParams(
        # in_channels=16,
        # base_filters=128 * 4,
        # kernel_size=(32, 16, 4, 2),
        # stride=4,
        # sigmoid=False,
        # output_size=None,
        # reinit=True,
        # # encoder_name="tf_efficientnet_b0.ns_jft_in1k",
        # encoder_name="tf_efficientnet_b2.ns_jft_in1k",
        # encoder_pretrained=True,
        in_channels=4,
        base_filters=128,
        kernel_size=(128, 64, 32, 16),
        stride=10000 // 512,  # time_step=10_000//strid
        # stride=4 * 4,  # time_step=10_000//strid
        # encoder_name="tf_efficientnet_b0.ns_jft_in1k",
        encoder_name="tf_efficientnet_b2.ns_jft_in1k",
        encoder_pretrained=True,
    )
    model = HMSCNNSpecFEModel(fe_params)
    # model = torch.compile(model, mode="default")
    bs = 4
    input = torch.rand(bs, 10000, fe_params.in_channels)
    print(f"{input.shape = }")
    bboxes, k_list, a_list = my_preprocess.prepare_cutmix(
        x_max=10000 // fe_params.stride, y_max=fe_params.base_filters, bs=bs
    )
    out = model(input, do_mixup=True, lam=0.5, y=torch.rand(bs, 6), bboxes=bboxes, k_list=k_list, a_list=a_list)
    print(f"{out['logits'].shape = }")


def _test() -> None:
    # _test_hms()
    _test_hms_transformer()
    # _test_hms_cnn_spec_fe()


if __name__ == "__main__":
    _test()
