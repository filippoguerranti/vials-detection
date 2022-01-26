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

from typing import List, Tuple, Union

import torch
import torchvision

from . import data, models, utils, cnn, autoencoder


def train(
    model: str,
    dataset_dir: str,
    classes: List[str],
    img_size: Tuple[int, int],
    splits: List[float],
    augmentation: bool,
    code_size: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    num_workers: int,
    device: str,
) -> None:

    # dataset preparation
    print("\nDataset preparation...")

    dataset = data.VialsDataset(dir=dataset_dir, classes=classes, imgs_size=img_size)

    ds_train, ds_val, ds_test = dataset.split(proportions=splits, shuffle=True)

    # apply data augmentation
    if augmentation:
        data_augmentation = utils.DataAugmentation(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.25,
            crop_resize=img_size,
            crop_scale=(0.8, 1),
            horizontal_flip_prob=0.6,
            gaussian_noise=(0, 0.01),
            gaussian_noise_prob=0.5,
        )
        ds_train.set_augmentation(data_augmentation)
        ds_val.set_augmentation(data_augmentation)

    # model preparation
    print("Model preparation...")

    # model = cnn.CNN(3, 40, len(classes), device=device)
    # model = autoencoder.SupervisedAutoencoder(
    #     n_channels=3, encoded_size=40, n_classes=4, device="cuda:0"
    # )
    # output = model(torch.unsqueeze(ds_train.__getitem__(0)[0], 0))[0]
    # target = torch.unsqueeze(ds_train.__getitem__(0)[1], 0).to("cuda:0")
    # print(target)
    # print(output)
    # print(torch.nn.functional.softmax(output))
    # print(torch.nn.CrossEntropyLoss()(output, target))

    if model.lower() == "ae":
        model = models.AE(
            code_size=code_size,
            device=device,
        )
        model.compile(optimizer="adam", lr=learning_rate, loss="mse", accuracy=False)

    elif model.lower() == "cnn":
        model = models.CNN(
            code_size=code_size,
            num_classes=len(classes),
            device=device,
        )
        model.compile(
            optimizer="adam", lr=learning_rate, loss="crossentropy", accuracy=True
        )

    elif model.lower() == "sae":
        model = models.SAE(
            num_classes=len(classes),
            code_size=code_size,
            device=device,
        )
        model.compile(optimizer="adam", lr=learning_rate, loss=None, accuracy=True)

    else:
        raise ValueError(f"unrecognized model {model.lower()}")

    # getting training data loader
    train_loader = ds_train.get_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        weighted_sampler=True,
        shuffle=False,
    )

    # getting validation data loader
    val_loader = ds_val.get_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        weighted_sampler=True,
        shuffle=False,
    )
    # model.train_cnn(
    #     train_loader, val_loader, batch_size, learning_rate, epochs, num_workers
    # )

    model.fit(
        training_loader=train_loader,
        validation_loader=val_loader,
        epochs=epochs,
    )

    # getting validation data loader
    test_loader = ds_test.get_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        weighted_sampler=False,
        shuffle=True,
    )

    test_loss, test_acc = model.evaluate(test_loader, best_model=True)

    message = f"\ntest_loss: {test_loss:.4f}" + (
        f" - test_acc: {test_acc:.4f}" if test_acc is not None else ""
    )
    print(message)


def classify(args):
    pass


def evaluate(
    model_path: str,
    dataset_dir: str,
    classes: List[str],
    splits: List[float],
) -> None:
    # dataset preparation
    print("\nDataset preparation...\n")

    dataset = data.VialsDataset(dir=dataset_dir, classes=classes)

    _, _, ds_test = dataset.split(proportions=splits, shuffle=False)

    ds_test.statistics()

    # model preparation
    print("\nModel preparation...\n")

    model_name = model_path.split("/")[-1]
    model = str(model_name.split("-")[0])
    code_size = int(model_name.split("-")[1])

    if model.lower() == "ae":
        model = models.Autoencoder(code_size=code_size)

    elif model.lower() == "cnn":
        model = models.CNN(code_size=code_size, num_classes=len(classes))

    elif model.lower() == "eae":
        model = models.EmpoweredAutoencoder(
            code_size=code_size, num_classes=len(classes)
        )

    else:
        raise ValueError(f"unrecognized model {model.lower()}")

    model.load(model_path)

    if getattr(model, "show_samples", False):
        import datetime

        model_name_no_ext = str(".".join(model_name.split(".")[:-1]))
        now = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
        SAVE_NAME = f"{model_name_no_ext}-{now}"
        model.show_samples(
            ds_test,
            indices=[0, 200, 201, 202, 3500, 3501, 3502, 4700, 5801, 5872],
            save=SAVE_NAME,
        )
