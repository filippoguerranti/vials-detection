import os
from typing import Any

import torch
import torch.nn as nn
from utils.definitions import ROOT_DIR

from .parts import Classifier, Encoder32, Encoder90

__all__ = ["CNN90", "CNN32", "cnn90", "cnn32"]

pretrained_path = {
    "cnn32-20": os.path.join(
        ROOT_DIR,
        r"checkpoints",
        r"best_cnn32-20.pth.tar",
    ),
    "cnn90-20": os.path.join(
        ROOT_DIR,
        r"checkpoints",
        r"best_cnn90-20.pth.tar",
    ),
    "cnn32-50": os.path.join(
        ROOT_DIR,
        r"checkpoints",
        r"best_cnn32-50.pth.tar",
    ),
    "cnn90-50": os.path.join(
        ROOT_DIR,
        r"checkpoints",
        r"best_cnn90-50.pth.tar",
    ),
    "cnn32-100": os.path.join(
        ROOT_DIR,
        r"checkpoints",
        r"best_cnn32-100.pth.tar",
    ),
    "cnn90-100": os.path.join(
        ROOT_DIR,
        r"checkpoints",
        r"best_cnn90-100.pth.tar",
    ),
}


class CNN90(nn.Module):
    """Convolutional Neural Network for 90x90 images.

    Composed of:
        - Encoder90
        - Classifier (2 double linear blocks)
    """

    def __init__(self, encoded_size: int, n_classes: int, n_channels: int = 3) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.encoded_size = encoded_size
        self.n_classes = n_classes

        # [3, 90, 90] = 24300
        self.encoder = Encoder90(n_channels, encoded_size)  # [encoded_size]
        self.classifier = Classifier(encoded_size, n_classes, blocks=2)  # [n_classes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        output = self.classifier(encoded)
        return output


class CNN32(torch.nn.Module):
    """Convolutional Neural Network for 32x32 images.

    Composed of:
        - Encoder32
        - Classifier (1 double linear blocks)
    """

    def __init__(self, encoded_size: int, n_classes: int, n_channels: int = 3) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.encoded_size = encoded_size
        self.n_classes = n_classes

        # [3, 32, 32] = 3072
        self.encoder = Encoder32(n_channels, encoded_size)  # [encoded_size]
        self.classifier = Classifier(encoded_size, n_classes, blocks=1)  # [n_classes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        output = self.classifier(encoded)
        return output


def cnn32(pretrained: bool = False, encoded_size: int = 20, **kwargs: Any) -> CNN32:
    model = CNN32(encoded_size=encoded_size, **kwargs)
    if pretrained:
        checkpoint = torch.load(pretrained_path[f"cnn32-{encoded_size}"])
        model.load_state_dict(checkpoint["state_dict"])
    return model


def cnn90(pretrained: bool = False, encoded_size: int = 20, **kwargs: Any) -> CNN90:
    model = CNN90(encoded_size=encoded_size, **kwargs)
    if pretrained:
        checkpoint = torch.load(pretrained_path[f"cnn90-{encoded_size}"])
        model.load_state_dict(checkpoint["state_dict"])
        print(
            f"\t=> model: {checkpoint['arch']} | epoch {checkpoint['epoch']} | best_mcc: {checkpoint['best_mcc']}"
        )
    return model
