import os
from typing import Any, Tuple

import torch
import torch.nn as nn
from utils.definitions import ROOT_DIR

from .parts import *

__all__ = ["SAE32", "SAE90", "sae32", "sae90"]

pretrained_path = {
    "sae32-20": os.path.join(
        ROOT_DIR,
        r"checkpoints",
        r"best_sae32-20.pth.tar",
    ),
    "sae90-20": os.path.join(
        ROOT_DIR,
        r"checkpoints",
        r"best_sae90-20.pth.tar",
    ),
    "sae32-50": os.path.join(
        ROOT_DIR,
        r"checkpoints",
        r"best_sae32-50.pth.tar",
    ),
    "sae90-50": os.path.join(
        ROOT_DIR,
        r"checkpoints",
        r"best_sae90-50.pth.tar",
    ),
    "sae32-100": os.path.join(
        ROOT_DIR,
        r"checkpoints",
        r"best_sae32-100.pth.tar",
    ),
    "sae90-100": os.path.join(
        ROOT_DIR,
        r"checkpoints",
        r"best_sae90-100.pth.tar",
    ),
}


class SAE90(nn.Module):
    """Supervised Autoencoder for 90x90 images.

    Composed of:
        - Encoder90
        - Decoder90
        - Classifier (2 double linear blocks)
    """

    def __init__(self, encoded_size: int, n_classes: int, n_channels: int = 3) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.encoded_size = encoded_size
        self.n_classes = n_classes

        # [3, 90, 90] = 24300
        self.encoder = Encoder90(n_channels, encoded_size)  # [encoded_size]
        self.decoder = Decoder90(encoded_size, n_channels)  # [3, 90, 90] = 24300
        self.classifier = Classifier(encoded_size, n_classes, blocks=2)  # [n_classes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output = self.classifier(encoded)
        return output, decoded


class SAE32(torch.nn.Module):
    """Supervised Autoencoder for 32x32 images.

    Composed of:
        - Encoder32
        - Decoder32
        - Classifier (1 double linear blocks)
    """

    def __init__(self, encoded_size: int, n_classes: int, n_channels: int = 3) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.encoded_size = encoded_size
        self.n_classes = n_classes

        # [3, 32, 32] = 3072
        self.encoder = Encoder32(n_channels, encoded_size)  # [encoded_size]
        self.decoder = Decoder32(encoded_size, n_channels)  # [3, 32, 32] = 3072
        self.classifier = Classifier(encoded_size, n_classes, blocks=1)  # [n_classes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output = self.classifier(encoded)
        return output, decoded


def sae32(pretrained: bool = False, encoded_size: int = 20, **kwargs: Any) -> SAE32:
    model = SAE32(encoded_size=encoded_size, **kwargs)
    if pretrained:
        checkpoint = torch.load(pretrained_path[f"sae32-{encoded_size}"])
        model.load_state_dict(checkpoint["state_dict"])
    return model


def sae90(pretrained: bool = False, encoded_size: int = 20, **kwargs: Any) -> SAE90:
    model = SAE90(encoded_size=encoded_size, **kwargs)
    if pretrained:
        checkpoint = torch.load(pretrained_path[f"sae90-{encoded_size}"])
        model.load_state_dict(checkpoint["state_dict"])
    return model
