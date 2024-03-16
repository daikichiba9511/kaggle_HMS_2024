from collections.abc import Callable

import torch
import torch.nn as nn
import torchaudio.transforms as TAT
from typing_extensions import Self


class Res1DBlock(nn.Module):
    def __init__(
        self: Self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        downsampling: nn.Module,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout_prob, inplace=False)
        self.conv1d1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv1d2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.downsampling = downsampling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv1d1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv1d2(x)

        out = self.pool(x)

        # skip connection
        identity = self.downsampling(identity)
        out += identity
        return out


class SELayer(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        # ndim=2/3 => 1d, ndim=4 => 2d
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[0], x.shape[1]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SERes1DBlock(nn.Module):
    def __init__(
        self: Self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        downsampling: nn.Module,
        reduction: int = 16,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout_prob, inplace=False)
        self.conv1d1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv1d2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.downsampling = downsampling
        self.se = SELayer(in_channels=out_channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv1d1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv1d2(x)
        out = self.pool(x)

        out = self.se(out.unsqueeze(-1))
        out = out.squeeze(-1)

        # skip connection
        identity = self.downsampling(identity)
        out += identity
        return out


class Parallel1DConvFeatureExtractor(nn.Module):
    def __init__(self: Self, kernels: list[int], in_channels: int = 20, fixed_kernel_size: int = 17) -> None:
        super().__init__()
        self.kernels = kernels
        self.planes = 24
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels

        for kernel_size in kernels:
            sep_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.planes,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=False,
            )
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1d1 = nn.Conv1d(
            in_channels=self.planes,
            out_channels=self.planes,
            kernel_size=fixed_kernel_size,
            stride=2,
            padding=2,
            bias=False,
        )

        self.res_block = self._make_resnet_layer(
            kernel_size=fixed_kernel_size, stride=1, padding=fixed_kernel_size // 2, base_width=self.planes
        )

        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.pool = nn.AvgPool1d(kernel_size=6, stride=6, padding=2)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """Forward to make embeddings.

        Args:
            x (torch.Tensor): (batch_size, in_channels, time_steps)

        Returns:
            out (torch.Tensor): (batch_size, planes * out.shape[-1])

        Notes:
                |-> conv1d1(ks_1) --|
            x-> |   ...             |--> concat -> feature_extract -> embeddings(shape=(bs, embed))
                |-> conv1d1(ks_N) --|

        References:
        1. https://www.kaggle.com/code/nischaydnk/lightning-1d-eegnet-training-pipeline-hbs?scriptVersionId=160814948
        2. https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/471666
        """
        out_sep = []
        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1d1(out)

        out = self.res_block(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.pool(out)

        # (bs, planes, out.shape[-1]) -> (bs, planes * out.shape[-1])
        out = out.reshape(x.size(0), -1)
        return out

    def _make_resnet_layer(
        self: Self, kernel_size: int, stride: int, padding: int, base_width: int, n_blocks: int = 9
    ) -> nn.Module:
        layers = []
        for _ in range(n_blocks):
            downsampling = nn.Sequential(nn.MaxPool1d(kernel_size=2, stride=2, padding=0))
            layers.append(
                Res1DBlock(
                    in_channels=base_width,
                    out_channels=base_width,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    downsampling=downsampling,
                )
            )
        return nn.Sequential(*layers)


class SpecNormalize(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize spectrogram.

        Args:
            x (torch.Tensor): (batch_size, in_channels, height=freq, width)

        Returns:
            torch.Tensor: (batch_size, in_channels, height, width)

        References:
        [1]
        https://github.com/tubo213/kaggle-child-mind-institute-detect-sleep-states/blob/main/src/models/feature_extractor/spectrogram.py
        """
        min_ = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_ = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        return (x - min_) / (max_ - min_ + self.eps)


class SpecFeatureExtractor(nn.Module):
    """Spectrogram feature extractor.

    References:
    [1]
    https://github.com/tubo213/kaggle-child-mind-institute-detect-sleep-states/blob/main/src/models/feature_extractor/spectrogram.py
    """

    def __init__(
        self,
        in_channels: int,
        height: int,
        hop_length: int,
        win_length: int | None = None,
        out_size: int | None = None,
    ) -> None:
        super().__init__()
        self.height = height
        self.out_chans = in_channels
        n_fft = (2 * height) - 1
        self.feature_extractor = nn.Sequential(
            TAT.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length),
            TAT.AmplitudeToDB(top_db=80),
            SpecNormalize(),
        )
        self.out_size = out_size
        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels - 1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img = self.feature_extractor(x)
        print(f"1: {img.shape=}")
        img = self.conv2d(img)
        print(f"2: {img.shape=}")
        if self.out_size is not None:
            img = self.pool(img)
        return img


class CNNSpectgram(nn.Module):
    """

    Refs:
    [1]
    https://github.com/analokmaus/kaggle-g2net-public/blob/main/models1d_pytorch/wavegram.py
    [2]
    https://github.com/tubo213/kaggle-child-mind-institute-detect-sleep-states/blob/main/src/models/feature_extractor/cnn.py

    """

    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int | tuple[int, ...] = 128,
        kernel_size: tuple[int, ...] = (32, 16, 4, 2),
        stride: int = 4,
        sigmoid: bool = False,
        output_size: int | None = None,
        conv: type[nn.Module] = nn.Conv1d,
        reinit: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_chans = len(kernel_size)
        self.out_size = output_size
        self.sigmoid = sigmoid

        if isinstance(base_filters, int):
            base_filters = tuple([base_filters])

        self.height = base_filters[-1]
        self.spec_conv = nn.ModuleList()
        for i in range(self.out_chans):
            tmp_block = [
                conv(
                    in_channels=in_channels,
                    out_channels=base_filters[0],
                    kernel_size=kernel_size[i],
                    stride=stride,
                    padding=(kernel_size[i] - 1) // 2,
                )
            ]

            if len(base_filters) > 1:
                for j in range(len(base_filters) - 1):
                    tmp_block = tmp_block + [
                        nn.BatchNorm1d(base_filters[j]),
                        nn.ReLU(inplace=True),
                        conv(
                            base_filters[j],
                            base_filters[j + 1],
                            kernel_size=kernel_size[i],
                            stride=stride,
                            padding=(kernel_size[i] - 1) // 2,
                        ),
                    ]
                self.spec_conv.append(nn.Sequential(*tmp_block))
            else:
                self.spec_conv.append(tmp_block[0])

        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

        if reinit:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward to make a spectrogram image.

        Args:
            x (torch.Tensor): (batch_size, in_channels, time_steps)

        Returns:
            torch.Tensor: (batch_size, out_chans, height, time_steps)
        """

        # x: (batch_size, in_channels, time_steps)
        out: list[torch.Tensor] = []
        for i in range(self.out_chans):
            # (batch_size, base_filters[i], time_steps // stride)
            spec_out = self.spec_conv[i](x)
            out.append(spec_out)

        # img: (batch_size, out_chans, height=base_filters, time_steps)
        img = torch.stack(out, dim=1)
        if self.out_size is not None:
            img = self.pool(img)  # (batch_size, out_size, height, out_size)
        if self.sigmoid:
            img = torch.sigmoid(img)
        return img


def _test_spec_feature_extractor() -> None:
    batch_size = 2
    in_channels = 4
    height = 64
    win_length = 32
    hop_length = 256
    # out_size = 64 * 2
    out_size = 24 * 60 * 4 // 2

    num_features = 4
    seq_len = 24 * 60 * 4

    x = torch.randn(batch_size, num_features, seq_len)
    fe = SpecFeatureExtractor(
        in_channels=in_channels,
        height=height,
        hop_length=hop_length,
        win_length=win_length,
        out_size=out_size,
    )
    print(f"{fe = }")
    print(f"{fe.out_chans = }")
    out = fe(x)
    print(f"{out.shape=}")
    assert out.shape == (
        batch_size,
        out_size,
        height,
        out_size,
    ), f"{out.shape=} != {(batch_size, num_features, height, out_size)}"


def _test_cnn_feature_extractor() -> None:
    batch_size = 2
    in_channels = 3
    num_features = 3
    seq_len = 24 * 60 * 4
    # kernel_size = (32, 16, 4, 2)
    # kernel_size = (15, 7, 4, 3, 1)  # 短期的な関係を見る
    kernel_size = (128, 64, 32, 16, 8, 5)  # 長期的な関係を見る
    # base_filters = (128, 256)
    # base_filters = (256, 160, 128)  # spectrogramの高さ、周波数領域の解像度
    base_filters = 128  # spectrogramの高さ、周波数領域の解像度
    out_channels = len(kernel_size)
    print(f"{batch_size = }, {num_features = }, {seq_len = }, {base_filters = }, {out_channels = }")

    x = torch.randn(batch_size, num_features, seq_len)
    print("input shape = ", x.shape)
    fe = CNNSpectgram(
        in_channels=in_channels, base_filters=base_filters, kernel_size=kernel_size, stride=4, reinit=True
    )
    print(f"{fe = }")
    out = fe(x)
    print(f"{out.shape = }")
    assert out.shape == (
        batch_size,
        out_channels,
        base_filters,
        1440,
    ), f"{out.shape=} != {(batch_size, out_channels, base_filters, 1440)}"


def _test_parallel_1dconv_fe() -> None:
    print("-- parallel 1dconv --")
    in_channels = 3
    kernels = [3, 5, 7, 9]

    input = torch.rand(2, in_channels, 2500)
    fe = Parallel1DConvFeatureExtractor(kernels=kernels, in_channels=in_channels, fixed_kernel_size=5)
    o = fe(input)
    print(o.shape)


def _test_seres1dblock() -> None:
    in_channels = 20
    out_channels = 10
    kernel_size = 3
    stride = 1
    padding = 1
    downsampling = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
    reduction = 16
    dropout_prob = 0.0
    input = torch.rand(2, in_channels, 2500)
    fe = SERes1DBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        downsampling=downsampling,
        reduction=reduction,
        dropout_prob=dropout_prob,
    )
    o = fe(input)
    print(o.shape)


if __name__ == "__main__":
    # _test_spec_feature_extractor()
    _test_cnn_feature_extractor()
    # _test_parallel_1dconv_fe()
    # _test_seres1dblock()
