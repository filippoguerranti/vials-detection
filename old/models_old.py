"""vials-detection/modules/models.py

Summary:
    Defines the models architectures.


Copyright 2021 - Filippo Guerranti <filippo.guerranti@student.unisi.it>

DO NOT REDISTRIBUTE. 
The software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY 
KIND, either express or implied.
"""

import datetime
import math
import os
import re
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import torch
import tqdm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)

from modules import data
from utils import utils


class Autoencoder(torch.nn.Module):
    """Autoencoder class.

    Convolutional autoencoder. The inputs are images of shape (32, 32, 3) and the outputs
    are images of shape (32, 32, 3).

    Autoencoder(
        (encoder): Encoder(
            (encoder_conv): Sequential(
                (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(2, 2))
                (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
                (2): ReLU(inplace=True)
                (3): Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
                (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
                (5): ReLU(inplace=True)
                (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
                (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
                (8): ReLU(inplace=True)
                (9): Flatten(start_dim=1, end_dim=-1)
                (10): Linear(in_features=256, out_features=100, bias=True)
                (11): ReLU(inplace=True)
                (12): Dropout(p=0.5, inplace=False)
            )
        )
        (decoder): Decoder(
            (decoder_lin): Sequential(
                (0): Linear(in_features=70, out_features=256, bias=True)
                (1): ReLU(inplace=True)
                (2): Unflatten(dim=1, unflattened_size=(64, 2, 2))
                (3): UpsamplingBilinear2d(scale_factor=2.5, mode=bilinear)
                (4): ConvTranspose2d(64, 32, kernel_size=(5, 5), stride=(2, 2), output_padding=(1, 1))
                (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
                (6): ReLU(inplace=True)
                (7): ConvTranspose2d(32, 3, kernel_size=(5, 5), stride=(2, 2), output_padding=(1, 1))
                (8): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True)
                (9): Sigmoid()
            )
        )
    )

    Args:
        code_size (int):
            Dimension of the encoded space. Default: 100.
        device (str):
            Device in which to load the model.
    """

    def __init__(
        self,
        code_size: int,
        device: str = "cpu",
    ) -> None:
        super(Autoencoder, self).__init__()

        # device setup
        self.device = (
            torch.device(device)
            if torch.cuda.is_available() and bool(re.findall("cuda:[\d]+$", device))
            else torch.device("cpu")
        )

        # model architecture
        self.code_size = code_size
        self.training_params = {"code_size": code_size}

        # [batch_size, 3, 32, 32]
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=0,
            ),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(inplace=True),
            # [batch_size, 32, 14, 14]
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=0,
            ),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=True),
            # [batch_size, 64, 5, 5]
            torch.nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=True),
            # [batch_size, 64, 2, 2]
            torch.nn.Flatten(start_dim=1),
            # [batch_size, 256]
            torch.nn.Linear(in_features=64 * 2 * 2, out_features=code_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
        )
        # [batch_size, code_size]

        # [batch_size, code_size]
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=code_size, out_features=64 * 2 * 2),
            torch.nn.ReLU(inplace=True),
            # [batch_size, 256]
            torch.nn.Unflatten(dim=1, unflattened_size=(64, 2, 2)),
            # [batch_size, 64, 2, 2]
            torch.nn.UpsamplingBilinear2d(scale_factor=2.5),
            torch.nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=0,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(inplace=True),
            # [batch_size, 32, 14, 14]
            torch.nn.ConvTranspose2d(
                in_channels=32,
                out_channels=3,
                kernel_size=5,
                stride=2,
                padding=0,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(num_features=3),
            torch.nn.Sigmoid(),
        )
        # [batch_size, 3, 32, 32]

        # move model to proper device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def save(self, path: str) -> None:
        """Save the model.

        The encoder and decoder parameters are saved to memory.

        Args:
            path (str):
                Path of the saved file, must have `.pth` extension
        """
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
            },
            path,
        )

    def load(self, path: str, **args) -> None:
        """Load the model.

        The encoder and decoder parameters are loaded from memory.

        Args:
            path (str)
                Path of the loaded file, must have `.pth` extension
            **args:
                (optional) parameters to pass to `torch.load_state_dict` function.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.to(self.device).load_state_dict(checkpoint["encoder"], **args)
        self.decoder.to(self.device).load_state_dict(checkpoint["decoder"], **args)

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the output of the network.

        The forward phase maps the input image into its encoded representation and the
        reconstructed image.

        Args:
            x (torch.Tensor):
                4D input tensor of the CNN [batch_size, 3, 32, 32].
            training (bool, optional):
                If True, the module is in training mode, otherwise it is in evalutation
                mode. Default: True.

        Returns:
            decoded (torch.Tensor):
                4D Tensor, the reconstructed image [batch_size, 3, 32, 32].
            encoded (torch.Tensor):
                2D Tensor, the encoded representation [batch_size, code_size].
        """
        training_mode_originally_on = self.encoder.training or self.decoder.training

        if training:
            self.__set_train_mode(True)
        else:
            self.__set_train_mode(False)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        if training_mode_originally_on:
            self.__set_train_mode(True)

        return decoded, encoded

    def __set_train_mode(self, mode: bool = True) -> torch.nn.Module:
        """Sets the module in training mode.

        This has any effect only on certain modules. See documentations of particular
        modules for details of their behaviors in training/evaluation mode, if they are
        affected, e.g. Dropout, BatchNorm, etc.

        Args:
            mode (bool):
                Whether to set training mode (True) or evaluation mode (False).
                Default: True.

        Returns:
            self (torch.nn.Module)
        """
        self.encoder.train(mode=mode)
        self.decoder.train(mode=mode)

    def __is_training(self) -> bool:
        return self.encoder.training or self.decoder.training

    def __loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the model.

        The given loss function (provided by the compile method) is used to compute the
        loss between the output and the expected target.

        Args:
            output (torch.Tensor):
                The output of the model, can have any dimension.
            target (torch.Tensor):
                The target associated to the model output, can have any dimensions in
                accordance with the dimensions of the output.

        Returns:
            loss (float):
                Value of the loss function.
        """

        return self.loss_fn(outputs, targets)

    def __metrics(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> Union[Dict, None]:

        if self.metrics is None:
            return None

        metrics = {metric: None for metric in self.metrics}
        pred_labels = torch.argmax(outputs, dim=1)
        labels = torch.argmax(targets, dim=1)

        for metric in self.metrics:

            if metric.lower() == "accuracy":
                m = accuracy_score(labels, pred_labels)

            if metric.lower() == "balanced_accuracy":
                m = balanced_accuracy_score(labels, pred_labels)

            if metric.lower() == "precision":
                m = precision_score(labels, pred_labels, average=self.metrics_average)

            if metric.lower() == "recall":
                m = recall_score(labels, pred_labels, average=self.metrics_average)

            metrics[metric] = m

        return metrics

    def compile(
        self,
        optimizer: str,
        lr: float,
        loss: Union[str, Tuple[str, str]],
        metrics: Union[str, List[str], None] = None,
        metrics_average: Union[str, None] = None,
    ) -> None:
        """Configures the model for training.

        Sets the optimizer, the learning rate, the loss function and the metrics.

        Args:
            optimizer (str):
                Optimizer. Possible choices are "Adam", "SGD", "RMSprop".
            lr (float):
                Learning rate for the optimizer.
            loss (Union[str, Tuple[str, str]]):
                Loss function. If a tuple is provided, the first value is the name of the
                loss and the second one is the reduction method.
                Possible choices for the loss are "MSELoss", "CrossEntropyLoss", "L1Loss",
                "MultiMarginLoss". Possible choices for the reduction are "mean", "sum".
            metrics (Union[str, List[str], None], optional):
                Metric(s) to be evaluated.
                Possible choices are "accuracy", "balanced_accuracy", "precision, "recall".
                Default: None.
            metrics_average (Union[str, None], optional):
                Determines the type of averaging performed on the metrics. Possible choices
                are:
                - "micro":
                    Calculate metrics globally by counting the total true positives, false
                    negatives and false positives.
                - "macro":
                    Calculate metrics for each label, and find their unweighted mean. This
                    does not take label imbalance into account.
                - "weighted":
                    Calculate metrics for each label, and find their average weighted by
                    support (the number of true instances for each label). This alters
                    "macro" to account for label imbalance; it can result in an F-score
                    that is not between precision and recall.
                Default: None.
        """

        # set optimizer
        params_to_optimize = [
            {"params": filter(lambda p: p.requires_grad, self.encoder.parameters())},
            {"params": filter(lambda p: p.requires_grad, self.decoder.parameters())},
        ]
        if isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                self.optimizer = torch.optim.Adam(
                    params=params_to_optimize,
                    lr=lr,
                )
            elif optimizer.lower() == "sgd":
                self.optimizer = torch.optim.SGD(
                    params=params_to_optimize,
                    lr=lr,
                )
            elif optimizer.lower() == "rmsprop":
                self.optimizer = torch.optim.RMSprop(
                    params=params_to_optimize,
                    lr=lr,
                )
            else:
                print(f"unrecognized optimizer: {optimizer}. Set to Adam.")
                self.optimizer = torch.optim.Adam(
                    params=params_to_optimize,
                    lr=lr,
                )
        else:
            raise ValueError(
                f"optimizer should be of type str. Instead {type(optimizer)}"
            )

        # set loss function
        if isinstance(loss, tuple):
            if not all([isinstance(l, str) for l in loss]):
                raise TypeError(
                    f"both loss name and reduction should be of type str. Instead: {[type(l) for l in loss]}"
                )
            loss_fn = loss[0]
            reduction = loss[1]
        elif isinstance(loss, str):
            loss_fn = loss
            reduction = "mean"
        else:
            raise TypeError(
                f"loss should be of type str or tuple(str, str). Instead {type(loss)}"
            )

        if loss_fn.lower() == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction=reduction)
        elif loss_fn.lower() == "crossentropy":
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
        elif loss_fn.lower() == "l1":
            self.loss_fn = torch.nn.L1Loss(reduction=reduction)
        elif loss_fn.lower() == "hinge":
            self.loss_fn = torch.nn.MultiMarginLoss(reduction=reduction)
        else:
            print(f"unrecognized loss: {loss}. Set to MSELoss.")
            self.loss_fn = torch.nn.MSELoss(reduction=reduction)

        # set metrics
        possible_metrics = ("accuracy", "balanced_accuracy", "precision", "recall")
        if isinstance(metrics, list):
            if not all([isinstance(m, str) for m in metrics]):
                raise TypeError(
                    f"metrics should be of type str. Instead: {[type(l) for l in metrics]}"
                )
            if not all([metric in possible_metrics for metric in metrics]):
                print(f"unrecognized metrics: {metrics}. Set to accuracy.")
                metrics = ["accuracy"]
        elif isinstance(metrics, str):
            metrics = [metrics]
        elif metrics is None:
            metrics = None
        else:
            raise TypeError(
                f"metrics should be of type str or List[str]. Instead {type(metrics)}"
            )
        self.metrics = metrics

        if isinstance(metrics_average, str):
            if metrics_average in (
                "micro",
                "macro",
                "weighted",
            ):
                self.metrics_average = metrics_average
            else:
                print(f"unrecognized metrics_average: {metrics_average}. Set to micro.")
                self.metrics_average = "micro"
        elif metrics_average is None:
            self.metrics_average = None
        else:
            raise TypeError(
                f"metrics_average should be of type str. Instead {type(metrics_average)}"
            )
        self.training_params.update({"lr": lr, "optimizer": optimizer, "loss": loss_fn})

    def fit(
        self,
        training_data: torch.utils.data.Dataset,
        validation_data: torch.utils.data.Dataset,
        batch_size: int = 64,
        epochs: int = 1,
        sampler: bool = True,
        model_checkpoint_folder: str = "vials-detection/models/",
        metric_check: Union[str, None] = None,
        num_workers: int = 3,
    ) -> None:
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        [extended_summary]

        Args:
            training_data (torch.utils.data.Dataset):
                Training dataset.
            validation_data (torch.utils.data.Dataset):
                Validation dataset.
            batch_size (int, optional):
                Number of examples per batch. Default: 64.
            epochs (int, optional):
                Number of training epochs. Default: 1.
            sampler (bool, optional):
                Determines if a WeightedRandomSampler will be used to load batches.
                Default: True.
            model_checkpoint_folder (str, optional):
                Path to the checkpoint folder. Default: "vials-detection/models/".
            metric_check (Union[str, None], optional):
                The metric to be checked for best model save. If None, the best model is
                saved according to the loss function. Default: None.
            num_workers (int, optional):
                Number of workers used to load the batches. Default: 3.
        """

        if not hasattr(self, "training_params"):
            raise RuntimeError(f"compile function should be called before fit.")

        self.training_params.update({"epochs": epochs, "batch_size": batch_size})

        self.model_name = self.__model_name()
        model_checkpoint_filepath = (
            f"{os.path.join(model_checkpoint_folder, self.model_name)}.pth"
        )

        if not os.path.exists(model_checkpoint_folder):
            os.makedirs(model_checkpoint_folder)

        # initialization
        self.__set_train_mode(True)

        best_epoch = -1

        best_validation_loss = math.inf
        self.epochs_validation_loss_list = list()
        self.epochs_training_loss_list = list()

        if self.metrics is not None:
            best_validation_metric = -1.0
            self.epochs_validation_metrics_list = list()
            self.epochs_training_metrics_list = list()
        else:
            best_validation_metric = None
            self.epochs_validation_metrics_list = None
            self.epochs_training_metrics_list = None

        # getting training data loaders
        train_loader = (
            training_data.get_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                weighted_sampler=True,
                shuffle=False,
            )
            if sampler
            else training_data.get_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                weighted_sampler=False,
                shuffle=True,
            )
        )

        # training over epochs
        for e in range(0, epochs):

            print(f"Epoch {e + 1}/{epochs}")

            epoch_training_loss = 0.0
            epoch_training_metrics = (
                {metric: None for metric in self.metrics}
                if self.metrics is not None
                else None
            )
            epoch_num_training_examples = 0

            # looping on batches
            for X, _ in tqdm.tqdm(train_loader):

                # if len(training_set) % batch_size != 0, then batch_size on last iteration is != from user batch_size
                batch_num_training_examples = X.shape[0]
                epoch_num_training_examples += batch_num_training_examples

                X = X.to(self.device)

                output, _ = self.forward(X)
                loss = self.__loss(output, X)

                self.optimizer.zero_grad()  # put all gradients to zero before computing backward phase
                loss.backward()  # computing gradients (for parameters with requires_grad=True)
                self.optimizer.step()  # updating parameters according to optimizer

                # mini-batch evaluation
                with torch.no_grad():  # keeping off the autograd engine

                    self.__set_train_mode(False)

                    if self.metrics is not None:
                        batch_training_metrics = self.__metrics(output, X)
                        epoch_training_metrics = {
                            metric: epoch_training_metrics.get(metric, 0)
                            + batch_training_metrics.get(batch_training_metrics, 0)
                            for metric in epoch_training_metrics.keys()
                        }

                    epoch_training_loss += loss.item() * batch_num_training_examples

                    self.__set_train_mode(True)

            # training set results
            epoch_training_loss /= epoch_num_training_examples
            self.epochs_training_loss_list.append(epoch_training_loss)

            if self.metrics is not None:
                epoch_training_metrics = {
                    metric: epoch_training_metrics.get(metric, 0)
                    / epoch_num_training_examples
                    for metric in epoch_training_metrics.keys()
                }
                self.epochs_training_metrics_list.append(epoch_training_metrics)
            else:
                self.epochs_training_metrics_list = None

            # validation set evaluation
            epoch_validation_loss, epoch_validation_metrics = self.evaluate(
                evaluation_data=validation_data,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
            )

            if epoch_validation_metrics is not None and metric_check is not None:
                if epoch_validation_metrics[metric_check] > best_validation_metric:
                    best_validation_metric = epoch_validation_metrics[metric_check]
                    best_epoch = e + 1
            else:
                if epoch_validation_loss < best_validation_loss:
                    best_validation_loss = epoch_validation_loss
                    best_epoch = e + 1

                # saving the best model so far
                self.save(path=model_checkpoint_filepath)

            self.epochs_validation_loss_list.append(epoch_validation_loss)

            if self.metrics is not None:
                epoch_validation_metrics = {
                    metric: epoch_validation_metrics.get(metric, 0)
                    / epoch_num_training_examples
                    for metric in epoch_training_metrics.keys()
                }
                self.epochs_validation_metrics_list.append(epoch_validation_metrics)
            else:
                self.epochs_validation_metrics_list = None

            message = (
                f"train_loss: {epoch_training_loss:.4f}"
                + (
                    " - ".join(
                        [
                            f"train_{metric}: {value:.4f}"
                            for metric, value in epoch_training_metrics.items()
                        ]
                    )
                    if self.metrics is not None
                    else ""
                )
                + f" - val_loss: {epoch_validation_loss:.4f}"
                + (
                    " - ".join(
                        [
                            f"val_{metric}: {value:.4f}"
                            for metric, value in epoch_validation_metrics.items()
                        ]
                    )
                    if self.metrics is not None
                    else ""
                )
                + (" - BEST!" if best_epoch == e + 1 else "")
            )

            print(message)

        self.__plot_training()

    def evaluate(
        self,
        evaluation_data: torch.utils.data.Dataset,
        batch_size: int = 64,
        sampler: bool = True,
        num_workers: int = 3,
    ) -> Union[float, List[float]]:
        """Returns the loss value & metrics values for the model in evaluation mode.

        [extended_summary]

        Args:
            evaluation_data (torch.utils.data.Dataset):
                Evaluation dataset.
            batch_size (int, optional):
                Number of examples per batch. Default: 64.
            sampler (bool, optional):
                Determines if a WeightedRandomSampler will be used to load batches.
                Default: True.
            num_workers (int, optional):
                Number of workers used to load the batches. Default: 3.

        Returns:
            loss (float):
                Loss of the network on evaluation dataset.
            metrics (List[float]):
                Metrics of the network on evaluation dataset.
        """

        # getting evaluation data loader
        evaluation_loader = (
            evaluation_data.get_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                weighted_sampler=True,
                shuffle=False,
            )
            if sampler
            else evaluation_data.get_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                weighted_sampler=False,
                shuffle=True,
            )
        )

        batch_outputs = []
        batch_inputs = []
        training_mode_originally_on = self.__is_training()

        with torch.no_grad():  # keeping off autograd engine

            if training_mode_originally_on:
                self.__set_train_mode(False)

            # mini-batch evaluation
            for X, _ in tqdm.tqdm(evaluation_loader):

                X = X.to(self.device)

                output, _ = self.forward(X)

                # append operation forced to be computed in cpu
                batch_outputs.append(output.cpu())
                batch_inputs.append(X.cpu())

            loss = self.__loss(
                torch.cat(batch_outputs, dim=0), torch.cat(batch_inputs, dim=0)
            )
            metrics = self.__metrics(
                torch.cat(batch_outputs, dim=0), torch.cat(batch_inputs, dim=0)
            )

            if training_mode_originally_on:
                self.__set_train_mode(True)

        return loss, metrics

        # def predict(self, input, batch_size):
        #     pass

        # def train_model(
        #     self,
        #     training_set: data.VialsDataset,
        #     validation_set: data.VialsDataset,
        #     batch_size: int = 64,
        #     lr: float = 0.001,
        #     epochs: int = 10,
        #     sampler: bool = True,  # if false, then shuffle
        #     num_workers: int = 3,
        #     model_path: str = "./models/",
        #     warmstart: Union[str, None] = None,
        # ) -> None:
        #     """Autoencoder training procedure.

        #     Does the following:
        #         - if `warmstart` is not None and it is a valid path to a model, then the model's
        #         parameters are used as an initialization for the training procedure.
        #         - sets the optimizer parameters (Adam is used as optimizer).
        #         - starts the training procedure over the epochs using the provided loss function
        #         to evaluate the performances of reconstruction of the input images on the validation
        #         set.
        #         - the model is saved if its performances on the validation set are better then the
        #         ones computed up to that point.

        #     Args:
        #         training_set (data.VialsDataset):
        #             Training dataset
        #         validation_set (data.VialsDataset):
        #             Validation dataset
        #         batch_size (int, optional):
        #             Number of examples per batch. Default: 64.
        #         lr (float, optional):
        #             Learning rate. Default: 0.001.
        #         epochs (int, optional):
        #             Number of training epochs. Default: 10.
        #         sampler (bool, optional):
        #             If True, the WeightedRandomSampler is used to load batches.
        #             Default: True.
        #         num_workers (int, optional):
        #             Number of workers used to load batches. Default: 3.
        #         model_path (str, optional):
        #             Path to the model saving folder. Default: "vials-detection/models/".
        #         warmstart (Union[str, None], optional):
        #             If it is not None and it is a valid path, the provided model is used as
        #             initializer for the training procedure. Default: None.
        #     """

        #     # model name for saving
        #     now = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
        #     base_name = f"{self.code_size}-{batch_size}b-{epochs}e-{lr}l-{now}"
        #     self.model_name = f"AE-{base_name}"
        #     FILEPATH = f"{os.path.join(model_path, self.model_name)}.pth"

        #     if not os.path.exists(model_path):
        #         os.makedirs(model_path)

        #     # warmstart initialization
        #     if (
        #         warmstart is not None
        #         and os.path.exists(warmstart)
        #         and os.path.isfile(warmstart)
        #         and warmstart.endswith(".pth")
        #     ):
        #         assert (
        #             warmstart.split("-")[-6] == self.code_size
        #         ), f"wrong model shape {warmstart.split('-')[-6]} vs. {self.code_size}"
        #         self.load(warmstart)

        #     # initialization
        #     self.__set_train_mode(True)

        #     best_validation_loss = math.inf  # best loss on the val data
        #     best_epoch = -1  # epoch in which best losses was computed
        #     self.epochs_validation_loss_list = list()  # list of epoch losses on val set
        #     self.epochs_training_loss_list = list()  # list of epoch losses on training set

        #     # getting training data loaders
        #     train_loader = (
        #         training_set.get_loader(
        #             batch_size=batch_size,
        #             num_workers=num_workers,
        #             weighted_sampler=True,
        #             shuffle=False,
        #         )
        #         if sampler
        #         else training_set.get_loader(
        #             batch_size=batch_size,
        #             num_workers=num_workers,
        #             weighted_sampler=False,
        #             shuffle=True,
        #         )
        #     )

        #     # set optimizer
        #     params_to_optimize = [
        #         {"params": filter(lambda p: p.requires_grad, self.encoder.parameters())},
        #         {"params": filter(lambda p: p.requires_grad, self.decoder.parameters())},
        #     ]
        #     self.optimizer = torch.optim.Adam(
        #         params=params_to_optimize,
        #         lr=lr,
        #     )

        #     # training over epochs
        #     for e in range(0, epochs):

        #         print(f"Epoch {e + 1}/{epochs}")

        #         epoch_training_loss = 0.0  # loss of current epoch over training set
        #         epoch_num_training_examples = (
        #             0  # accumulated number of training examples for current epoch
        #         )

        #         # looping on batches
        #         for image, _ in tqdm.tqdm(train_loader):

        #             # if len(training_set) % batch_size != 0, then batch_size on last iteration is != from user batch_size
        #             batch_num_training_examples = image.shape[0]
        #             epoch_num_training_examples += batch_num_training_examples

        #             image = image.to(self.device)

        #             decoded, _ = self.forward(image)
        #             loss = Autoencoder.__loss(decoded, image)

        #             self.optimizer.zero_grad()  # put all gradients to zero before computing backward phase
        #             loss.backward()  # computing gradients (for parameters with requires_grad=True)
        #             self.optimizer.step()  # updating parameters according to optimizer

        #             # mini-batch evaluation
        #             with torch.no_grad():  # keeping off the autograd engine

        #                 self.__set_train_mode(False)

        #                 epoch_training_loss += loss.item() * batch_num_training_examples

        #                 self.__set_train_mode(True)

        #         # validation set evaluation
        #         epoch_validation_loss = self.eval_model(
        #             dataset=validation_set,
        #             batch_size=batch_size,
        #             num_workers=num_workers,
        #             sampler=sampler,
        #         )

        #         if epoch_validation_loss < best_validation_loss:
        #             best_validation_loss = epoch_validation_loss
        #             best_epoch = e + 1

        #             # saving the best model so far
        #             self.save(path=FILEPATH)

        #         self.epochs_validation_loss_list.append(epoch_validation_loss)

        #         epoch_training_loss /= epoch_num_training_examples
        #         self.epochs_training_loss_list.append(epoch_training_loss)

        #         print(
        #             f"loss: {epoch_training_loss:.4f} - val_loss: {epoch_validation_loss:.4f}"
        #             + (" - BEST!" if best_epoch == e + 1 else "")
        #         )

        #     self.__plot_training()

        # def eval_model(
        #     self,
        #     dataset: data.VialsDataset,
        #     batch_size: int = 64,
        #     num_workers: int = 3,
        #     sampler: bool = True,
        # ) -> float:
        # """Autoencoder evaluation procedure.

        # Evaluates the losses over the provided dataset by forwarding it (batch by batch)
        # through the model and accumulating the losses on each mini-batch.

        # Args:
        #     dataset (data.VialsDataset):
        #         Dataset to be evaluated.
        #     batch_size (int, optional):
        #         Number of examples per batch. Default: 64.
        #     num_workers (int, optional):
        #         Number of workers used to load batches. Default: 3.
        #     sampler (bool, optional):
        #         If True, the WeightedRandomSampler is used to load batches.
        #         Default: True.

        # Returns:
        #     loss (float):
        #         Loss of the network on input dataset.
        # """

        # # getting validation data loader
        # dataset_loader = (
        #     dataset.get_loader(
        #         batch_size=batch_size,
        #         num_workers=num_workers,
        #         weighted_sampler=True,
        #         shuffle=False,
        #     )
        #     if sampler
        #     else dataset.get_loader(
        #         batch_size=batch_size,
        #         num_workers=num_workers,
        #         weighted_sampler=False,
        #         shuffle=True,
        #     )
        # )

        # batch_outputs = []
        # batch_images = []
        # training_mode_originally_on = self.encoder.training or self.decoder.training

        # with torch.no_grad():  # keeping off autograd engine

        #     if training_mode_originally_on:
        #         self.__set_train_mode(False)

        #     # mini-batch evaluation
        #     for image, _ in tqdm.tqdm(dataset_loader):

        #         image = image.to(self.device)

        #         encoded = self.encoder(image)
        #         decoded = self.decoder(encoded)

        #         # append operation forced to be computed in cpu
        #         batch_outputs.append(decoded.cpu())
        #         batch_images.append(image.cpu())

        #     loss = Autoencoder.__loss(
        #         torch.cat(batch_outputs, dim=0), torch.cat(batch_images, dim=0)
        #     )

        #     if training_mode_originally_on:
        #         self.__set_train_mode(True)

        # return loss

    def __plot_training(self) -> None:
        """Plots validation and training losses over the epochs."""

        # parameters retrieval
        code_size = self.training_params.get("code_size", None)
        batch_size = self.training_params.get("batch_size", None)
        epochs = self.training_params.get("epochs", None)
        optimizer = self.training_params.get("optimizer", None)
        lr = self.training_params.get("lr", None)
        loss = self.training_params.get("loss", None)

        # plotting
        x = list(range(1, epochs + 1))
        plt.plot(x, self.epochs_training_loss_list, label="Training")
        plt.plot(x, self.epochs_validation_loss_list, label="Validation")
        plt.grid(True)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        title = (
            f"code_size={code_size}"
            + f", batch_size={batch_size}"
            + f", epochs={epochs}"
            + f", optimizer={optimizer}"
            + f", lr={lr}"
            + f", loss={loss}"
        )
        plt.title(title)
        plt.legend()

        # save plot
        BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        FOLDER = os.path.join(BASEDIR, "results/")
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)
        FILEPATH = os.path.join(FOLDER, self.model_name)
        plt.savefig(f"{FILEPATH}.png", dpi=1000)

    def show_samples(
        self,
        dataset: data.VialsDataset,
        num_images: Union[int, None] = None,
        indices: Union[List[int], None] = None,
        save: bool = False,
    ) -> None:
        """Show input images, encoded representation and decoded images.

        Uses the model to encode the input images from dataset and decode them.
        The inputs and the outputs of the model are shown alongside with the encoded
        representations, so that a visual comparison can be made.

        Args:
            dataset (data.VialsDataset):
                Dataset from which to sample input images.
            num_images (Union[int, None], optional):
                Number of images to compare. If None, indices must be provided.
                Default: None.
            indices (Union[List[int], None]):
                Indices of the images to be shown. If None, random images are selected.
                Default: None.
            save (bool, optional):
                Save the model in the folder "results/images" under the name model_name_num_images
                or model_name_indices.
        """

        if num_images is not None:
            dataset_loader = dataset.get_loader(
                batch_size=num_images, num_workers=3, weighted_sampler=True
            )
            for images, _ in dataset_loader:
                original = images.to(self.device)
                break
            MODEL_NAME = self.model_name + f"{num_images}images"

        elif indices is not None:
            original = torch.cat(
                [torch.unsqueeze(dataset[i][0], 0) for i in indices]
            ).to(self.device)
            num_images = len(indices)
            MODEL_NAME = (
                self.model_name + f"({'-'.join(str(i) for i in indices)})indices"
            )

        else:
            raise ValueError(
                f"num_images and indices cannot be provided simultaneously: {num_images}, {indices}"
            )

        training_mode_originally_on = self.__is_training()

        with torch.no_grad():
            if training_mode_originally_on:
                self.__set_train_mode(False)
            decoded, encoded = self.forward(original)
            if training_mode_originally_on:
                self.__set_train_mode(True)

        original = original.cpu()
        encoded = encoded.cpu()
        code_size = encoded.shape[1]
        decoded = decoded.cpu()

        plt.figure(figsize=(10, 6))
        for i in range(num_images):

            original_ = original[i][0].numpy()
            ax = plt.subplot(3, num_images, i + 1)
            plt.imshow(original_, cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == num_images // 2:
                ax.set_title("Original images")

            h, w = utils.closest_factors(code_size)
            encoded_ = encoded[i].numpy().reshape(h, w)
            ax = plt.subplot(3, num_images, i + 1 + num_images)
            plt.imshow(encoded_, cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == num_images // 2:
                ax.set_title("Encoded representation")

            decoded_ = decoded[i][0].numpy()
            ax = plt.subplot(3, num_images, i + 1 + (num_images * 2))
            plt.imshow(decoded_, cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == num_images // 2:
                ax.set_title("Reconstructed images")
        fig = plt.gcf()
        plt.show()

        # save plot
        if save:
            BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            FOLDER = os.path.join(BASEDIR, "results/images")
            if not os.path.exists(FOLDER):
                os.makedirs(FOLDER)
            FILEPATH = os.path.join(FOLDER, MODEL_NAME)
            fig.savefig(f"{FILEPATH}.png", dpi=1000)

    def __model_name(self):

        if not hasattr(self, "training_params"):
            raise RuntimeError(f"compile function should be called before fit.")

        now = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
        code_size = self.code_size
        batch_size = self.training_params.get("batch_size", None)
        epochs = self.training_params.get("epochs", None)
        optimizer = self.training_params.get("optimizer", None)
        lr = self.training_params.get("lr", None)
        loss = self.training_params.get("loss", None)
        base_name = (
            f"{code_size}-{batch_size}b-{epochs}e-{optimizer}-{lr}l-{loss}-{now}"
        )
        model_name = f"AE-{base_name}"

        return model_name


class CNN(torch.nn.Module):
    def __init__(
        self,
        code_size: int,
        num_classes: int,
        load: Union[str, None] = None,
        device: str = "cpu",
    ) -> None:
        super(CNN, self).__init__()

        # device setup
        self.device = (
            torch.device(device)
            if torch.cuda.is_available() and bool(re.findall("cuda:[\d]+$", device))
            else torch.device("cpu")
        )

        # model architecture
        self.code_size = code_size
        self.num_classes = num_classes

        # [batch_size, 3, 32, 32]
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=12,
                kernel_size=7,
                stride=1,
                padding="same",
            ),
            torch.nn.BatchNorm2d(num_features=12),
            torch.nn.ReLU(inplace=True),
            # [batch_size, 12, 32, 32]
            torch.nn.Conv2d(
                in_channels=12,
                out_channels=24,
                kernel_size=5,
                stride=1,
                padding="same",
            ),
            torch.nn.BatchNorm2d(num_features=24),
            torch.nn.ReLU(inplace=True),
            # [batch_size, 24, 32, 32]
            torch.nn.Conv2d(
                in_channels=24,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(inplace=True),
            # [batch_size, 32, 32, 32]
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=0,
            ),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=True),
            # [batch_size, 64, 14, 14]
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=0,
            ),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=True),
            # [batch_size, 64, 5, 5]
            torch.nn.Flatten(start_dim=1),
            # [batch_size, 1600]
            torch.nn.Linear(in_features=64 * 5 * 5, out_features=self.code_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
        )
        # [batch_size, code_size]
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.code_size, out_features=self.num_classes)
        )
        # [batch_size, num_classes]

        # load model
        if (
            load is not None
            and os.path.exists(load)
            and os.path.isfile(load)
            and load.endswith(".pth")
        ):
            self.load(load)

        # move model to proper device
        self.cnn.to(self.device)
        self.output_layer.to(self.device)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the output of the network.

        The forward phase generates a tensor of non-normalized predictions (logits) which
        is then passed to a normalization function (softmax) in order to obtain the proper
        outputs. Moreover, the tensor of the encoded representation is returned.

        Args:
            x (torch.Tensor):
                4D input tensor of the CNN [batch_size, 3, 32, 32].

        Returns:
            outputs (torch.Tensor):
                2D tensor, normalized outputs (after softmax act.) [batch_size, 10].
            logits (torch.Tensor):
                2D tensor non-normalized outputs (before softmax act.) [batch_size, 10].
            encoded (torch.Tensor):
                2D tensor, encoded representation [batch_size, code_size].
        """
        encoded = self.cnn(x)
        logits = self.output_layer(encoded)
        outputs = torch.nn.functional.softmax(logits, dim=1)

        return outputs, logits, encoded

    def __set_train_mode(self, mode: bool = True) -> torch.nn.Module:
        """Sets the module in training mode.

        This has any effect only on certain modules. See documentations of particular
        modules for details of their behaviors in training/evaluation mode, if they are
        affected, e.g. Dropout, BatchNorm, etc.

        Args:
            mode (bool):
                Whether to set training mode (True) or evaluation mode (False).
                Default: True.

        Returns:
            self (torch.nn.Module)
        """
        self.cnn.train(mode=mode)
        self.output_layer.train(mode=mode)

    @staticmethod
    def __loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the loss function of the CNN.

        The logits (non-normalized outputs) are used to compute the Cross-Entropy loss.

        Args:
            logits (torch.Tensor):
                2D output Tensor of the net (before softmax act.) [batch_size, 10].
            labels (torch.Tensor):
                1D label Tensor of the outputs [batch_size].
            weight (torch.Tensor):
                1D weight Tensor of the classes distribution, usefull in unbalanced
                datasets. The classes having less samples will be weighted more w.r.t.
                the more populated ones.

        Returns:
            loss (float):
                Value of the loss function.
        """
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

        return loss_fn(logits, labels)

    @staticmethod
    def __accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Computes the prediction accuracy.

        The accuracy is computed as num_correct_predictions / num_total_predictions.

        Args:
            outputs (torch.Tensor)
                2D output Tensor of the net (after softmax act.) [batch_size, 10].
            labels (torch.Tensor)
                1D label Tensor of the outputs [batch_size].

        Returns:
            accuracy (float)
                Prediction accuracy.
        """
        decisions = torch.argmax(outputs, dim=1)
        targets = torch.argmax(labels, dim=1)

        correct_decisions = torch.eq(decisions, targets)  # element-wise equality
        accuracy = torch.mean(correct_decisions.float() * 100.0).item()

        return accuracy

    def save(self, path: str) -> None:
        """Save the model.

        The encoder and decoder parameters are saved to the provided path as `.pth` file.

        Args:
            path (str):
                Path of the saved file, must have `.pth` extension
        """
        torch.save(
            {
                "cnn": self.cnn.state_dict(),
                "output_layer": self.output_layer.state_dict(),
            },
            path,
        )

    def load(self, path: str, **args) -> None:
        """Load the model.

        The encoder and decoder parameters are loaded from the provided `.pth` file.

        Args:
            path (str)
                Path of the loaded file, must have `.pth` extension
            **args:
                (optional) parameters to pass to `torch.load_state_dict` function.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.cnn.to(self.device).load_state_dict(checkpoint["cnn"], **args)
        self.output_layer.to(self.device).load_state_dict(
            checkpoint["output_layer"], **args
        )

    def train_model(
        self,
        training_set: data.VialsDataset,
        validation_set: data.VialsDataset,
        batch_size: int = 64,
        lr: float = 0.001,
        epochs: int = 10,
        sampler: bool = True,  # if false, then shuffle
        num_workers: int = 3,
        model_path: str = "vials-detection/models/",
        warmstart: Union[str, None] = None,
    ) -> None:
        """CNN training procedure.

        Functioning:
            - if `warmstart` is not None and it is a valid path to a model, then the model's
            parameters are used as an initialization for the training procedure.
            - sets the optimizer parameters (Adam is used as optimizer).
            - starts the training procedure over the epochs using the provided loss function
            to evaluate the performances of reconstruction of the input images on the validation
            set.
            - the model is saved if its performances on the validation set are better then the
            ones computed up to that point.

        Args:
            training_set (data.VialsDataset):
                Training dataset
            validation_set (data.VialsDataset):
                Validation dataset
            batch_size (int, optional):
                Number of examples per batch. Default: 64.
            lr (float, optional):
                Learning rate. Default: 0.001.
            epochs (int, optional):
                Number of training epochs. Default: 10.
            sampler (bool, optional):
                If True, the WeightedRandomSampler is used to load batches.
                Default: True.
            num_workers (int, optional):
                Number of workers used to load batches. Default: 3.
            model_path (str, optional):
                Path to the model saving folder. Default: "vials-detection/models/".
            warmstart (Union[str, None], optional):
                If it is not None and it is a valid path, the provided model is used as
                initializer for the training procedure. Default: None.
        """

        # model name for saving
        now = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
        base_name = f"{self.code_size}-{batch_size}b-{epochs}e-{lr}l-{now}"
        self.model_name = f"CNN-{base_name}"
        FILEPATH = f"{os.path.join(model_path, self.model_name)}.pth"

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # warmstart initialization
        if (
            warmstart is not None
            and os.path.exists(warmstart)
            and os.path.isfile(warmstart)
            and warmstart.endswith(".pth")
        ):
            assert (
                warmstart.split("-")[-6] == self.code_size
            ), f"wrong model shape {warmstart.split('-')[-6]} vs. {self.code_size}"
            self.load(warmstart)

        # initialization
        self.__set_train_mode(True)

        best_validation_acc = -1.0  # best accuracy on the val data
        best_epoch = -1  # epoch in which best losses was computed
        self.epochs_validation_loss_list = list()  # list of epoch losses on val set
        self.epochs_validation_acc_list = list()  # list of epoch accuracies on val set
        self.epochs_training_loss_list = list()  # list of epoch losses on training set
        self.epochs_training_acc_list = list()  # list of epoch losses on training set

        # getting training data loaders
        train_loader = (
            training_set.get_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                weighted_sampler=True,
                shuffle=False,
            )
            if sampler
            else training_set.get_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                weighted_sampler=False,
                shuffle=True,
            )
        )

        # set optimizer
        params_to_optimize = [
            {"params": filter(lambda p: p.requires_grad, self.cnn.parameters())},
            {
                "params": filter(
                    lambda p: p.requires_grad, self.output_layer.parameters()
                )
            },
        ]
        self.optimizer = torch.optim.Adam(
            params=params_to_optimize,
            lr=lr,
        )

        # training over epochs
        for e in range(0, epochs):

            print(f"Epoch {e + 1}/{epochs}")

            epoch_training_loss = 0.0  # loss of current epoch over training set
            epoch_training_acc = 0.0  # accuracy of current epoch over training set
            epoch_num_training_examples = (
                0  # accumulated number of training examples for current epoch
            )

            # looping on batches
            for X, Y in tqdm.tqdm(train_loader):

                # if len(training_set) % batch_size != 0, then batch_size on last iteration is != from user batch_size
                batch_num_training_examples = X.shape[0]
                epoch_num_training_examples += batch_num_training_examples

                X = X.to(self.device)
                Y = Y.to(self.device)

                output, logits, _ = self.forward(X)

                loss = CNN.__loss(logits, Y)

                self.optimizer.zero_grad()  # put all gradients to zero before computing backward phase
                loss.backward()  # computing gradients (for parameters with requires_grad=True)
                self.optimizer.step()  # updating parameters according to optimizer

                # mini-batch evaluation
                with torch.no_grad():  # keeping off the autograd engine

                    self.__set_train_mode(False)

                    batch_training_acc = CNN.__accuracy(output, Y)
                    epoch_training_acc += (
                        batch_training_acc * batch_num_training_examples
                    )

                    epoch_training_loss += loss.item() * batch_num_training_examples

                    self.__set_train_mode(True)

            # validation set evaluation
            epoch_validation_loss, epoch_validation_acc = self.eval_model(
                dataset=validation_set,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=sampler,
            )

            if epoch_validation_acc > best_validation_acc:
                best_validation_acc = epoch_validation_acc
                best_epoch = e + 1

                # saving the best model so far
                self.save(path=FILEPATH)

            self.epochs_validation_loss_list.append(epoch_validation_loss)
            self.epochs_validation_acc_list.append(epoch_validation_acc)

            epoch_training_loss /= epoch_num_training_examples
            epoch_training_acc /= epoch_num_training_examples
            self.epochs_training_loss_list.append(epoch_training_loss)
            self.epochs_training_acc_list.append(epoch_training_acc)

            print(
                f"train_loss: {epoch_training_loss:.4f} - train_acc: {epoch_training_acc:.4f} - val_loss: {epoch_validation_loss:.4f} - val_acc: {epoch_validation_acc:.4f}"
                + (" - BEST!" if best_epoch == e + 1 else "")
            )

        self.__plot_training()

    def eval_model(
        self,
        dataset: data.VialsDataset,
        batch_size: int = 64,
        num_workers: int = 3,
        sampler: bool = True,
    ) -> float:
        """CNN evaluation procedure.

        Evaluates the accuracy over the provided dataset by forwarding it (batch by batch)
        through the model and accumulating the accuracies on each mini-batch.

        Args:
            dataset (data.VialsDataset):
                Dataset to be evaluated.
            loss_fn (torch.nn.Loss, optional):
                Loss function. Default: torch.nn.MSELoss().
            batch_size (int, optional):
                Number of examples per batch. Default: 64.
            num_workers (int, optional):
                Number of workers used to load batches. Default: 3.
            sampler (bool, optional):
                If True, the WeightedRandomSampler is used to load batches.
                Default: True.

        Returns:
            loss (float):
                Loss of the network on input dataset.
        """

        # getting validation data loader
        dataset_loader = (
            dataset.get_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                weighted_sampler=True,
                shuffle=False,
            )
            if sampler
            else dataset.get_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                weighted_sampler=False,
                shuffle=True,
            )
        )

        batch_outputs = []
        batch_images = []
        training_mode_originally_on = self.cnn.training or self.output_layer.training

        with torch.no_grad():  # keeping off autograd engine

            if training_mode_originally_on:
                self.__set_train_mode(False)

            # mini-batch evaluation
            for X, Y in tqdm.tqdm(dataset_loader):

                X = X.to(self.device)

                outputs, _, _ = self.forward(X)

                # append operation forced to be computed in cpu
                batch_outputs.append(outputs.cpu())
                batch_images.append(Y.cpu())

            loss = CNN.__loss(
                torch.cat(batch_outputs, dim=0), torch.cat(batch_images, dim=0)
            )
            accuracy = CNN.__accuracy(
                torch.cat(batch_outputs, dim=0), torch.cat(batch_images, dim=0)
            )

            if training_mode_originally_on:
                self.__set_train_mode(True)

        return loss, accuracy

    def __plot_training(self) -> None:
        """Plots validation and training accuracies and losses over the epochs."""

        # parameters retrieval
        # CNN-CODE_SIZE-BATCH_SIZEb-EPOCHSe-LRl-DAY-TIME
        fields = self.model_name.split("-")
        # ['CNN', 'CODE_SIZE', 'BATCH_SIZEb', 'EPOCHSe', 'LRl', 'DAY', 'TIME']
        code_size = int(fields[1])
        batch_size = int(fields[2][:-1])
        epochs = int(fields[3][:-1])
        lr = float(fields[4][:-1])

        # plotting
        x = list(range(1, epochs + 1))

        fig, ax1 = plt.subplots()
        p1 = ax1.plot(x, self.epochs_training_loss_list, "-b", label="Training loss")
        p2 = ax1.plot(
            x, self.epochs_validation_loss_list, "--b", label="Validation loss"
        )
        ax2 = ax1.twinx()
        p3 = ax2.plot(x, self.epochs_training_acc_list, "-r", label="Training acc")
        p4 = ax2.plot(x, self.epochs_validation_acc_list, "--r", label="Validation acc")

        ax1.set_title(
            f"code_size={code_size}, batch_size={batch_size}, epochs={epochs}, lr={lr}"
        )
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax2.set_ylabel("Accuracy")
        ax1.grid()

        labs = [p.get_label() for p in p1 + p2 + p3 + p4]
        ax1.legend(p1 + p2 + p3 + p4, labs, loc=0)

        # save plot
        BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        FOLDER = os.path.join(BASEDIR, "results/")
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)
        FILEPATH = os.path.join(FOLDER, self.model_name)
        plt.savefig(f"{FILEPATH}.png", dpi=1000)

    def show_samples(
        self,
        dataset: data.VialsDataset,
        num_images: Union[int, None] = None,
        indices: Union[List[int], None] = None,
        save: Union[str, None] = None,
    ) -> None:
        """Show input images, encoded representation and decoded images.

        Uses the model to encode the input images from dataset and decode them.
        The inputs and the outputs of the model are shown alongside with the encoded
        representations, so that a visual comparison can be made.

        Args:
            dataset (data.VialsDataset):
                Dataset from which to sample input images.
            num_images (Union[int, None], optional):
                Number of images to compare. If None, indices must be provided.
                Default: None.
            indices (Union[List[int], None]):
                Indices of the images to be shown. If None, random images are selected.
                Default: None.
            save (Union[str, None], optional):
                If not None and a name is provided, the image is saved to "results/images"
                folder under the provided name. Default: None.
        """

        if not (num_images is None or indices is None):
            raise ValueError(
                f"num_images and indices cannot be provided simultaneously: {num_images}, {indices}"
            )

        if num_images is not None:
            dataset_loader = dataset.get_loader(
                batch_size=num_images, num_workers=3, weighted_sampler=True
            )
            for images, _ in dataset_loader:
                original = images.to(self.device)
                break

        elif indices is not None:
            original = torch.cat(
                [torch.unsqueeze(dataset[i][0], 0) for i in indices]
            ).to(self.device)
            num_images = len(indices)

        training_mode_originally_on = self.cnn.training or self.output_layer.training

        with torch.no_grad():
            if training_mode_originally_on:
                self.__set_train_mode(False)
            _, _, encoded = self.forward(original)
            if training_mode_originally_on:
                self.__set_train_mode(True)

        original = original.cpu()
        encoded = encoded.cpu()
        code_size = encoded.shape[1]

        plt.figure(figsize=(10, 6))
        for i in range(num_images):

            original_ = original[i][0].numpy()
            ax = plt.subplot(2, num_images, i + 1)
            plt.imshow(original_, cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == num_images // 2:
                ax.set_title("Original images")

            h, w = utils.closest_factors(code_size)
            encoded_ = encoded[i].numpy().reshape(h, w)
            ax = plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(encoded_, cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == num_images // 2:
                ax.set_title("Encoded representation")
        fig = plt.gcf()
        plt.show()

        # save plot
        if save is not None and isinstance(save, str):
            BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            FOLDER = os.path.join(BASEDIR, "results/images")
            if not os.path.exists(FOLDER):
                os.makedirs(FOLDER)
            FILEPATH = os.path.join(FOLDER, save)
            fig.savefig(f"{FILEPATH}.png", dpi=1000)


class EmpoweredAutoencoder(torch.nn.Module):
    def __init__(
        self,
        code_size: int,
        num_classes: int,
        load: Union[str, None] = None,
        device: str = "cpu",
    ) -> None:
        super(EmpoweredAutoencoder, self).__init__()

        # device setup
        self.device = (
            torch.device(device)
            if torch.cuda.is_available() and bool(re.findall("cuda:[\d]+$", device))
            else torch.device("cpu")
        )

        # model architecture
        self.code_size = code_size
        self.num_classes = num_classes

        # [batch_size, 3, 32, 32]
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=0,
            ),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(inplace=True),
            # [batch_size, 32, 14, 14]
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=0,
            ),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=True),
            # [batch_size, 64, 5, 5]
            torch.nn.MaxPool2d(
                kernel_size=2,
                stride=2,
                padding=0,
            ),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=True),
            # [batch_size, 64, 2, 2]
            torch.nn.Flatten(start_dim=1),
            # [batch_size, 256]
            torch.nn.Linear(in_features=64 * 2 * 2, out_features=code_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
        )
        # [batch_size, code_size]

        # [batch_size, code_size]
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=code_size, out_features=64 * 2 * 2),
            torch.nn.ReLU(inplace=True),
            # [batch_size, 256]
            torch.nn.Unflatten(dim=1, unflattened_size=(64, 2, 2)),
            # [batch_size, 64, 2, 2]
            torch.nn.UpsamplingBilinear2d(scale_factor=2.5),
            torch.nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=0,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(inplace=True),
            # [batch_size, 32, 14, 14]
            torch.nn.ConvTranspose2d(
                in_channels=32,
                out_channels=3,
                kernel_size=5,
                stride=2,
                padding=0,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(num_features=3),
            torch.nn.Sigmoid(),
        )
        # [batch_size, 3, 32, 32]

        # [batch_size, code_size]
        self.fully_connected = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.code_size,
                out_features=self.code_size * 2,
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            # [batch_size, code_size * 2]
            torch.nn.Linear(
                in_features=self.code_size * 2, out_features=self.num_classes
            ),
        )
        # [batch_size, num_classes]

        # load model
        if (
            load is not None
            and os.path.exists(load)
            and os.path.isfile(load)
            and load.endswith(".pth")
        ):
            self.load(load)

        # move model to proper device
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.fully_connected.to(self.device)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the output of the network.

        The forward phase generates a tensor of non-normalized predictions (logits) which
        is then passed to a normalization function (softmax) in order to obtain the proper
        outputs. Moreover, the tensor of the encoded representations is returned,
        alongside with the decoded images (reconstructed).

        Args:
            x (torch.Tensor):
                4D input tensor of the CNN [batch_size, 3, 32, 32].

        Returns:
            outputs (torch.Tensor):
                2D tensor, normalized outputs (after softmax act.) [batch_size, 10].
            logits (torch.Tensor):
                2D tensor non-normalized outputs (before softmax act.) [batch_size, 10].
            decoded (torch.Tensor):
                4D tensor, decoded representation [batch_size, 3, 32, 32].
            encoded (torch.Tensor):
                2D tensor, encoded representation [batch_size, code_size].
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        logits = self.fully_connected(encoded)
        outputs = torch.nn.functional.softmax(logits, dim=1)

        return outputs, logits, decoded, encoded

    def __set_train_mode(self, mode: bool = True) -> torch.nn.Module:
        """Sets the module in training mode.

        This has any effect only on certain modules. See documentations of particular
        modules for details of their behaviors in training/evaluation mode, if they are
        affected, e.g. Dropout, BatchNorm, etc.

        Args:
            mode (bool):
                Whether to set training mode (True) or evaluation mode (False).
                Default: True.

        Returns:
            self (torch.nn.Module)
        """
        self.encoder.train(mode=mode)
        self.decoder.train(mode=mode)
        self.fully_connected.train(mode=mode)

    @staticmethod
    def __loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        reconstructed_images: torch.Tensor,
        original_images: torch.Tensor,
        supervision_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
    ) -> torch.Tensor:
        """Compute the loss function of the EmpoweredAutoencoder.

        The logits (non-normalized outputs) are used to compute the Cross-Entropy loss.
        The reconstructed image is compared to the original one by mean of the MSE loss.
        The default reduction is "sum" for the MSE loss and "mean" for the CE loss.

        Args:
            logits (torch.Tensor):
                2D output Tensor of the net (before softmax act.) [batch_size, 10].
            labels (torch.Tensor):
                1D label Tensor of the outputs [batch_size].
            reconstructed_images (torch.Tensor):
                4D tensor, the output image (output of decoder) [batch_size, 3, 32, 32].
            original_images (torch.Tensor):
                4D tensor, the original image [batch_size, 3, 32, 32].
            supervision_weight (float):
                How much the supervision part (cross-entropy) should be weighted.
                Default: 1.
            reconstruction_weight (float):
                How much the reconstruction part (mse) should be weighted.
                Default: 1.

        Returns:
            loss (float):
                Value of the loss function.
        """

        mse = torch.nn.MSELoss(reduction="sum")
        ce = torch.nn.CrossEntropyLoss(reduction="mean")

        return supervision_weight * ce(logits, labels) + reconstruction_weight * mse(
            reconstructed_images, original_images
        )

    @staticmethod
    def __accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Computes the prediction accuracy.

        The accuracy is computed as num_correct_predictions / num_total_predictions.

        Args:
            outputs (torch.Tensor)
                2D output Tensor of the net (after softmax act.) [batch_size, 10].
            labels (torch.Tensor)
                1D label Tensor of the outputs [batch_size].

        Returns:
            accuracy (float)
                Prediction accuracy.
        """
        decisions = torch.argmax(outputs, dim=1)
        targets = torch.argmax(labels, dim=1)

        correct_decisions = torch.eq(decisions, targets)  # element-wise equality
        accuracy = torch.mean(correct_decisions.float() * 100.0).item()

        return accuracy

    def save(self, path: str) -> None:
        """Save the model.

        The encoder and decoder parameters are saved to the provided path as `.pth` file.

        Args:
            path (str):
                Path of the saved file, must have `.pth` extension
        """
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "fully_connected": self.fully_connected.state_dict(),
            },
            path,
        )

    def load(self, path: str, **args) -> None:
        """Load the model.

        The encoder and decoder parameters are loaded from the provided `.pth` file.

        Args:
            path (str)
                Path of the loaded file, must have `.pth` extension
            **args:
                (optional) parameters to pass to `torch.load_state_dict` function.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.encoder.to(self.device).load_state_dict(checkpoint["encoder"], **args)
        self.decoder.to(self.device).load_state_dict(checkpoint["decoder"], **args)
        self.fully_connected.to(self.device).load_state_dict(
            checkpoint["fully_connected"], **args
        )

    def train_model(
        self,
        training_set: data.VialsDataset,
        validation_set: data.VialsDataset,
        batch_size: int = 64,
        lr: float = 0.001,
        epochs: int = 10,
        sampler: bool = True,  # if false, then shuffle
        num_workers: int = 3,
        model_path: str = "vials-detection/models/",
        warmstart: Union[str, None] = None,
        supervision_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
    ) -> None:
        """EmpoweredAutoencoder training procedure.

        Functioning:
            - if `warmstart` is not None and it is a valid path to a model, then the model's
            parameters are used as an initialization for the training procedure.
            - sets the optimizer parameters (Adam is used as optimizer).
            - starts the training procedure over the epochs using the provided loss function
            to evaluate the performances of reconstruction of the input images on the validation
            set.
            - the model is saved if its performances on the validation set are better then the
            ones computed up to that point.

        Args:
            training_set (data.VialsDataset):
                Training dataset
            validation_set (data.VialsDataset):
                Validation dataset
            batch_size (int, optional):
                Number of examples per batch. Default: 64.
            lr (float, optional):
                Learning rate. Default: 0.001.
            epochs (int, optional):
                Number of training epochs. Default: 10.
            sampler (bool, optional):
                If True, the WeightedRandomSampler is used to load batches.
                Default: True.
            num_workers (int, optional):
                Number of workers used to load batches. Default: 3.
            model_path (str, optional):
                Path to the model saving folder. Default: "vials-detection/models/".
            warmstart (Union[str, None], optional):
                If it is not None and it is a valid path, the provided model is used as
                initializer for the training procedure. Default: None.
            supervision_weight (float):
                How much the supervision part (cross-entropy) should be weighted.
                Default: 1.
            reconstruction_weight (float):
                How much the reconstruction part (mse) should be weighted.
                Default: 1.
        """

        # model name for saving
        now = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
        base_name = f"{self.code_size}-{batch_size}b-{epochs}e-{lr}l-{now}"
        self.model_name = f"EAE-{base_name}"
        FILEPATH = f"{os.path.join(model_path, self.model_name)}.pth"

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # warmstart initialization
        if (
            warmstart is not None
            and os.path.exists(warmstart)
            and os.path.isfile(warmstart)
            and warmstart.endswith(".pth")
        ):
            assert (
                warmstart.split("-")[-6] == self.code_size
            ), f"wrong model shape {warmstart.split('-')[-6]} vs. {self.code_size}"
            self.load(warmstart)

        # initialization
        self.__set_train_mode(True)

        best_validation_acc = -1.0  # best accuracy on the val data
        best_epoch = -1  # epoch in which best losses was computed
        self.epochs_validation_loss_list = list()  # list of epoch losses on val set
        self.epochs_validation_acc_list = list()  # list of epoch accuracies on val set
        self.epochs_training_loss_list = list()  # list of epoch losses on training set
        self.epochs_training_acc_list = list()  # list of epoch losses on training set

        # getting training data loaders
        train_loader = (
            training_set.get_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                weighted_sampler=True,
                shuffle=False,
            )
            if sampler
            else training_set.get_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                weighted_sampler=False,
                shuffle=True,
            )
        )

        # set optimizer
        params_to_optimize = [
            {"params": filter(lambda p: p.requires_grad, self.encoder.parameters())},
            {"params": filter(lambda p: p.requires_grad, self.decoder.parameters())},
            {
                "params": filter(
                    lambda p: p.requires_grad, self.fully_connected.parameters()
                )
            },
        ]
        self.optimizer = torch.optim.Adam(
            params=params_to_optimize,
            lr=lr,
        )

        # training over epochs
        for e in range(0, epochs):

            print(f"Epoch {e + 1}/{epochs}")

            epoch_training_loss = 0.0  # loss of current epoch over training set
            epoch_training_acc = 0.0  # accuracy of current epoch over training set
            epoch_num_training_examples = (
                0  # accumulated number of training examples for current epoch
            )

            # looping on batches
            for X, Y in tqdm.tqdm(train_loader):

                # if len(training_set) % batch_size != 0, then batch_size on last iteration is != from user batch_size
                batch_num_training_examples = X.shape[0]
                epoch_num_training_examples += batch_num_training_examples

                X = X.to(self.device)
                Y = Y.to(self.device)

                output, logits, decoded, _ = self.forward(X)

                loss = EmpoweredAutoencoder.__loss(
                    logits=logits,
                    labels=Y,
                    reconstructed_images=decoded,
                    original_images=X,
                    supervision_weight=supervision_weight,
                    reconstruction_weight=reconstruction_weight,
                )

                self.optimizer.zero_grad()  # put all gradients to zero before computing backward phase
                loss.backward()  # computing gradients (for parameters with requires_grad=True)
                self.optimizer.step()  # updating parameters according to optimizer

                # mini-batch evaluation
                with torch.no_grad():  # keeping off the autograd engine

                    self.__set_train_mode(False)

                    batch_training_acc = EmpoweredAutoencoder.__accuracy(output, Y)
                    epoch_training_acc += (
                        batch_training_acc * batch_num_training_examples
                    )

                    epoch_training_loss += loss.item() * batch_num_training_examples

                    self.__set_train_mode(True)

            # validation set evaluation
            epoch_validation_loss, epoch_validation_acc = self.eval_model(
                dataset=validation_set,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=sampler,
            )

            if epoch_validation_acc > best_validation_acc:
                best_validation_acc = epoch_validation_acc
                best_epoch = e + 1

                # saving the best model so far
                self.save(path=FILEPATH)

            self.epochs_validation_loss_list.append(epoch_validation_loss)
            self.epochs_validation_acc_list.append(epoch_validation_acc)

            epoch_training_loss /= epoch_num_training_examples
            epoch_training_acc /= epoch_num_training_examples
            self.epochs_training_loss_list.append(epoch_training_loss)
            self.epochs_training_acc_list.append(epoch_training_acc)

            print(
                f"train_loss: {epoch_training_loss:.4f} - train_acc: {epoch_training_acc:.4f} - val_loss: {epoch_validation_loss:.4f} - val_acc: {epoch_validation_acc:.4f}"
                + (" - BEST!" if best_epoch == e + 1 else "")
            )

        self.__plot_training()

    def eval_model(
        self,
        dataset: data.VialsDataset,
        batch_size: int = 64,
        num_workers: int = 3,
        sampler: bool = True,
    ) -> float:
        """EmpoweredAutoencoder evaluation procedure.

        Evaluates the accuracy over the provided dataset by forwarding it (batch by batch)
        through the model and accumulating the accuracies on each mini-batch.

        Args:
            dataset (data.VialsDataset):
                Dataset to be evaluated.
            loss_fn (torch.nn.Loss, optional):
                Loss function. Default: torch.nn.MSELoss().
            batch_size (int, optional):
                Number of examples per batch. Default: 64.
            num_workers (int, optional):
                Number of workers used to load batches. Default: 3.
            sampler (bool, optional):
                If True, the WeightedRandomSampler is used to load batches.
                Default: True.

        Returns:
            loss (float):
                Loss of the network on input dataset.
        """

        # getting validation data loader
        dataset_loader = (
            dataset.get_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                weighted_sampler=True,
                shuffle=False,
            )
            if sampler
            else dataset.get_loader(
                batch_size=batch_size,
                num_workers=num_workers,
                weighted_sampler=False,
                shuffle=True,
            )
        )

        batch_outputs = []
        batch_decoded = []
        batch_labels = []
        batch_images = []
        training_mode_originally_on = (
            self.encoder.training
            or self.decoder.training
            or self.fully_connected.training
        )

        with torch.no_grad():  # keeping off autograd engine

            if training_mode_originally_on:
                self.__set_train_mode(False)

            # mini-batch evaluation
            for X, Y in tqdm.tqdm(dataset_loader):

                X = X.to(self.device)

                outputs, _, decoded, _ = self.forward(X)

                # append operation forced to be computed in cpu
                batch_outputs.append(outputs.cpu())
                batch_decoded.append(decoded.cpu())
                batch_labels.append(Y.cpu())
                batch_images.append(X.cpu())

            loss = EmpoweredAutoencoder.__loss(
                torch.cat(batch_outputs, dim=0),
                torch.cat(batch_labels, dim=0),
                torch.cat(batch_decoded, dim=0),
                torch.cat(batch_images, dim=0),
            )
            accuracy = EmpoweredAutoencoder.__accuracy(
                torch.cat(batch_outputs, dim=0), torch.cat(batch_labels, dim=0)
            )

            if training_mode_originally_on:
                self.__set_train_mode(True)

        return loss, accuracy

    def __plot_training(self) -> None:
        """Plots validation and training accuracies and losses over the epochs."""

        # parameters retrieval
        # EAE-CODE_SIZE-BATCH_SIZEb-EPOCHSe-LRl-DAY-TIME
        fields = self.model_name.split("-")
        # ['EAE', 'CODE_SIZE', 'BATCH_SIZEb', 'EPOCHSe', 'LRl', 'DAY', 'TIME']
        code_size = int(fields[1])
        batch_size = int(fields[2][:-1])
        epochs = int(fields[3][:-1])
        lr = float(fields[4][:-1])

        # plotting
        x = list(range(1, epochs + 1))

        fig, ax1 = plt.subplots()
        p1 = ax1.plot(x, self.epochs_training_loss_list, "--b", label="Training loss")
        p2 = ax1.plot(
            x, self.epochs_validation_loss_list, "--r", label="Validation loss"
        )
        ax2 = ax1.twinx()
        p3 = ax2.plot(x, self.epochs_training_acc_list, "-b", label="Training acc")
        p4 = ax2.plot(x, self.epochs_validation_acc_list, "-r", label="Validation acc")

        ax1.set_title(
            f"code_size={code_size}, batch_size={batch_size}, epochs={epochs}, lr={lr}"
        )
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax2.set_ylabel("Accuracy")
        ax1.grid()

        labs = [p.get_label() for p in p1 + p2 + p3 + p4]
        ax1.legend(p1 + p2 + p3 + p4, labs, loc=0)

        # save plot
        BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        FOLDER = os.path.join(BASEDIR, "results/")
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)
        FILEPATH = os.path.join(FOLDER, self.model_name)
        plt.savefig(f"{FILEPATH}.png", dpi=1000)

    def show_samples(
        self,
        dataset: data.VialsDataset,
        num_images: Union[int, None] = None,
        indices: Union[List[int], None] = None,
        save: Union[str, None] = None,
    ) -> None:
        """Show input images, encoded representation and decoded images.

        Uses the model to encode the input images from dataset and decode them.
        The inputs and the outputs of the model are shown alongside with the encoded
        representations, so that a visual comparison can be made.

        Args:
            dataset (data.VialsDataset):
                Dataset from which to sample input images.
            num_images (Union[int, None], optional):
                Number of images to compare. If None, indices must be provided.
                Default: None.
            indices (Union[List[int], None]):
                Indices of the images to be shown. If None, random images are selected.
                Default: None.
            save (Union[str, None], optional):
                If not None and a name is provided, the image is saved to "results/images"
                folder under the provided name. Default: None.
        """

        if not (num_images is None or indices is None):
            raise ValueError(
                f"num_images and indices cannot be provided simultaneously: {num_images}, {indices}"
            )

        if num_images is not None:
            dataset_loader = dataset.get_loader(
                batch_size=num_images, num_workers=3, weighted_sampler=True
            )
            for images, _ in dataset_loader:
                original = images.to(self.device)
                break

        elif indices is not None:
            original = torch.cat(
                [torch.unsqueeze(dataset[i][0], 0) for i in indices]
            ).to(self.device)
            num_images = len(indices)

        training_mode_originally_on = (
            self.encoder.training
            or self.decoder.training
            or self.fully_connected.training
        )

        with torch.no_grad():
            if training_mode_originally_on:
                self.__set_train_mode(False)
            _, _, decoded, encoded = self.forward(original)
            if training_mode_originally_on:
                self.__set_train_mode(True)

        original = original.cpu()
        encoded = encoded.cpu()
        code_size = encoded.shape[1]
        decoded = decoded.cpu()

        plt.figure(figsize=(10, 6))
        for i in range(num_images):

            original_ = original[i][0].numpy()
            ax = plt.subplot(3, num_images, i + 1)
            plt.imshow(original_, cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == num_images // 2:
                ax.set_title("Original images")

            h, w = utils.closest_factors(code_size)
            encoded_ = encoded[i].numpy().reshape(h, w)
            ax = plt.subplot(3, num_images, i + 1 + num_images)
            plt.imshow(encoded_, cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == num_images // 2:
                ax.set_title("Encoded representation")

            decoded_ = decoded[i][0].numpy()
            ax = plt.subplot(3, num_images, i + 1 + (num_images * 2))
            plt.imshow(decoded_, cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == num_images // 2:
                ax.set_title("Reconstructed images")
        fig = plt.gcf()
        plt.show()

        # save plot
        if save is not None and isinstance(save, str):
            BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            FOLDER = os.path.join(BASEDIR, "results/images")
            if not os.path.exists(FOLDER):
                os.makedirs(FOLDER)
            FILEPATH = os.path.join(FOLDER, save)
            fig.savefig(f"{FILEPATH}.png", dpi=1000)
