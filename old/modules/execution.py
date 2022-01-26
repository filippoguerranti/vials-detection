"""vials-detection/modules/execution.py

Summary:
    Defines the execution modes functions.

Functions:
    train()
    classify()
    evaluate()


Copyright 2021 - Filippo Guerranti <filippo.guerranti@student.unisi.it>

DO NOT REDISTRIBUTE. 
The software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY 
KIND, either expressed or implied.
"""

import collections
import os
import datetime
from typing import List, Tuple

import torch
import torchmetrics
import torchsummary
import tqdm
import math

from ..utils import utils

from . import data, models


def train(
    model: str,
    dataset_dir: str,
    classes: List[str],
    splits: List[float],
    augmentation: bool,
    test_augmentation: bool,
    encoded_size: int,
    reconstruction_weight: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    num_workers: int,
    device: str,
    datetime_: str,
) -> None:

    # dataset preparation
    print("\nDataset preparation...")

    dataset = data.VialsDataset(dir=dataset_dir, classes=classes)

    ds_train, ds_valid, ds_test = dataset.split(proportions=splits, shuffle=False)

    # apply data augmentation
    if augmentation:
        data_augmentation = utils.DataAugmentation(
            degrees=(-5.0, 5.0),
            crop_resize=(90, 90),
            crop_scale=(0.7, 0.9),
            brightness=(0.8, 1.5),
            contrast=(0.7, 1.8),
            saturation=(0.2, 1.8),
            hue=(-0.5, 0.5),
            horizontal_flip_prob=0.5,
            gaussian_noise=(0.0, 0.01),
            gaussian_noise_prob=0.5,
        )
        ds_train.set_augmentation(data_augmentation)
        ds_valid.set_augmentation(data_augmentation)
        if test_augmentation:
            ds_test.set_augmentation(data_augmentation)

    # getting training data loader
    train_loader = ds_train.get_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        weighted_sampler=True,
        shuffle=False,
    )

    # getting validation data loader
    valid_loader = ds_valid.get_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        weighted_sampler=True,
        shuffle=False,
    )

    # model preparation
    print("Model preparation...")
    models_main_folder = r"vials-detection/models"
    MODEL_NAME = model.upper()
    model_main_folder = os.path.join(models_main_folder, f"{MODEL_NAME}")
    if not os.path.exists(model_main_folder):
        os.makedirs(model_main_folder)
    model_entire_name = (
        f"{MODEL_NAME}"
        + f"-C{encoded_size}"
        + f"-RW{reconstruction_weight}"
        + f"-B{batch_size}"
        + f"-E{epochs}"
        + f"-LR{learning_rate}"
        + (f"-A{True}" if augmentation else f"-A{False}")
        + (f"-TA{True}" if test_augmentation else f"-TA{False}")
        + f"-{datetime_}"
    )
    model_folder = os.path.join(model_main_folder, model_entire_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    MODEL_PATH = os.path.join(
        model_folder,
        model_entire_name + ".pth",
    )

    if MODEL_NAME == "CNN":
        model = models.ConvolutionalNetwork(
            n_channels=3,
            encoded_size=encoded_size,
            n_classes=len(classes),
            device=device,
        )
        criterion = torch.nn.CrossEntropyLoss()

    elif MODEL_NAME == "SAE":
        model = models.SupervisedAutoencoder(
            n_channels=3,
            encoded_size=encoded_size,
            n_classes=len(classes),
            device=device,
        )
        criterion = utils.SAELoss(reconstruction_weight=reconstruction_weight)
    else:
        raise ValueError(f"unrecognized model {MODEL_NAME}")

    torchsummary.summary(model, (3, 32, 32))

    print("Start training...")
    model.compile(criterion=criterion, optimizer="adam", learning_rate=learning_rate)
    model.fit(train_loader, valid_loader, epochs, MODEL_PATH)

    ######################################################################################
    # TEST EVALUATION - start

    print("\n\nEvaluating...")

    # getting test data loader
    test_loader = ds_test.get_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        weighted_sampler=False,
        shuffle=False,
    )

    (
        test_loss,
        test_mcc,
        test_accuracy,
        test_f1score,
        test_precision,
        test_recall,
    ) = model.evaluate(test_loader, model_path=MODEL_PATH)

    message = (
        f"TEST | loss: {test_loss:.4f}"
        + f" - mcc: {test_mcc:.4f}"
        + f" - accuracy: {test_accuracy:.4f}"
        + f" - f1score: {test_f1score:.4f}"
        + f" - precision: {test_precision:.4f}"
        + f" - recall: {test_recall:.4f}"
        # + f" - precision: ["
        # + " ".join([f"{a.item():.4f}" for a in test_precision])
        # + "]"
    )
    print(message)
