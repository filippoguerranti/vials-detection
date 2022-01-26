from typing import Tuple, Union

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Double convolution module.

    Composed of:
        - conv2d
        - batchnorm2d
        - relu
        - conv2d
        - batchnorm2d
        - relu
        - dropout2d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        mid_channels: Union[int, None] = None,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = (out_channels + in_channels) // 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscale with maxpool2d and double convolution.

    Composed of:
        - maxpool2d
        - doubleconv
    """

    def __init__(self, in_channels: int, out_channels: int, conv_kernel_size: int = 3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(
                in_channels,
                out_channels,
                kernel_size=conv_kernel_size,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscale with upsample and double convolution.

    Composed of:
        - upsample
        - doubleconv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_kernel_size: int = 3,
        pad: bool = False,
    ):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size=conv_kernel_size)

        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if self.pad:
            # add padding to match dimension
            x = nn.functional.pad(x, [0, 1, 0, 1])

        return self.conv(x)


class DoubleLinear(nn.Module):
    """Double linear module

    Composed of:
        - linear
        - relu
        - linear
        - relu
        - dropout (if passed as parameter)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mid_features: Union[int, None] = None,
        dropout: bool = True,
    ) -> None:
        super().__init__()
        if mid_features is None:
            mid_features = (out_features + in_features) // 2
        modules = [
            nn.Linear(in_features, mid_features),
            nn.ReLU(inplace=True),
            nn.Linear(mid_features, out_features),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            modules.append(nn.Dropout(0.4))
        self.double_linear = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.double_linear(x)
        return x


class Classifier(nn.Module):
    """Classifier.

    Composed of:
        - linear
        - doublelinear (x num_blocks)
        - linear
    """

    def __init__(self, encoded_size: int, n_classes: int, blocks: int = 1) -> None:
        super().__init__()

        modules = (
            [nn.Linear(encoded_size, encoded_size * 2)]
            + [
                DoubleLinear(
                    encoded_size * 2, encoded_size * 2, encoded_size * 2, dropout=True
                )
                for _ in range(blocks)
            ]
            + [nn.Linear(encoded_size * 2, n_classes)]
        )
        # [encoded_size]
        self.linears = nn.Sequential(*modules)
        # [n_classes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linears(x)


class Encoder90(nn.Module):
    """Encoder that handles 90x90 images.

    Composed of:
        - doubleconv
        - down
        - down
        - down
        - down
        - flatten
        - doublelinear
    """

    def __init__(self, n_channels: int, encoded_size: int) -> None:
        super().__init__()

        # [3, 90, 90]
        self.dconv_in = DoubleConv(n_channels, 16, kernel_size=7)  # [16, 90, 90]
        self.down1 = Down(16, 32, conv_kernel_size=5)  # [32, 45, 45]
        self.down2 = Down(32, 64, conv_kernel_size=5)  # [64, 22, 22]
        self.down3 = Down(64, 128, conv_kernel_size=3)  # [128, 11, 11]
        self.down4 = Down(128, 256, conv_kernel_size=3)  # [256, 5, 5]
        self.flatten = nn.Flatten()  # [6400]
        self.linear_in = DoubleLinear(6400, encoded_size)  # [encoded_size]
        # [encoded_size]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dconv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.flatten(x)
        encoded = self.linear_in(x)

        return encoded


class Encoder32(nn.Module):
    """Encoder that handles 32x32 images.

    Composed of:
        - doubleconv
        - down
        - down
        - down
        - flatten
        - doublelinear
    """

    def __init__(self, n_channels: int, encoded_size: int) -> None:
        super().__init__()

        # [3, 32, 32]
        self.dconv_in = DoubleConv(n_channels, 16, kernel_size=5)  # [16, 32, 32]
        self.down1 = Down(16, 32, conv_kernel_size=5)  # [32, 16, 16]
        self.down2 = Down(32, 64, conv_kernel_size=3)  # [64, 8, 8]
        self.down3 = Down(64, 128, conv_kernel_size=3)  # [128, 4, 4]
        self.flatten = nn.Flatten()  # [2048]
        self.linear_in = DoubleLinear(2048, encoded_size)  # [encoded_size]
        # [encoded_size]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dconv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.flatten(x)
        encoded = self.linear_in(x)

        return encoded


class Decoder90(nn.Module):
    """Decoder that handles 90x90 images.

    Composed of:
        - doubleconv
        - unflatten
        - up
        - up
        - up
        - up
        - doubleconv
    """

    def __init__(self, encoded_size: int, n_channels: int) -> None:
        super().__init__()

        # [encoded_size]
        self.linear_out = DoubleLinear(encoded_size, 6400)  # [6400]
        self.unflatten = nn.Unflatten(1, (256, 5, 5))  # [256, 5, 5]
        self.up1 = Up(256, 128, conv_kernel_size=3, pad=True)  # [128, 11, 11]
        self.up2 = Up(128, 64, conv_kernel_size=3, pad=False)  # [64, 22, 22]
        self.up3 = Up(64, 32, conv_kernel_size=5, pad=True)  # [32, 45, 45]
        self.up4 = Up(32, 16, conv_kernel_size=5, pad=False)  # [16, 90, 90]
        self.dconv_out = DoubleConv(16, n_channels, kernel_size=7)  # [3, 90, 90]
        # [3, 90, 90]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_out(x)
        x = self.unflatten(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        decoded = self.dconv_out(x)
        return decoded


class Decoder32(nn.Module):
    """Decoder that handles 32x32 images.

    Composed of:
        - doubleconv
        - unflatten
        - up
        - up
        - up
        - doubleconv
    """

    def __init__(self, encoded_size: int, n_channels: int) -> None:
        super().__init__()

        # [encoded_size]
        self.linear_out = DoubleLinear(encoded_size, 2048)  # [2048]
        self.unflatten = nn.Unflatten(1, (128, 4, 4))  # [128, 4, 4]
        self.up1 = Up(128, 64, conv_kernel_size=3, pad=False)  # [64, 8, 8]
        self.up2 = Up(64, 32, conv_kernel_size=3, pad=False)  # [32, 16, 16]
        self.up3 = Up(32, 16, conv_kernel_size=5, pad=False)  # [16, 32, 32]
        self.dconv_out = DoubleConv(16, n_channels, kernel_size=5)  # [3, 32, 32]
        # [3, 32, 32]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_out(x)
        x = self.unflatten(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        decoded = self.dconv_out(x)

        return decoded
