"""vials-detection/modules/models.py

Summary:
    Defines the models architectures.


Copyright 2021 - Filippo Guerranti <filippo.guerranti@student.unisi.it>

DO NOT REDISTRIBUTE. 
The software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY 
KIND, either expressed or implied.
"""

import datetime
import math
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torchvision
import tqdm

from modules import data
from utils import utils


class Encoder(torch.nn.Module):
    def __init__(self, input_channels: int, code_size: int) -> None:
        """Encoder class.

        Implement a convolutional encoder that takes an input image of input_channels
        channels and maps it into a 1D vector of size code_size.

        Args:
            input_channels (int):
                Number of channels of input tensor/image.
            code_size (int):
                Size of the encoded representation.
        """
        super().__init__()

        # [batch_size, input_channels, 32, 32]
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=12,
                kernel_size=7,
                stride=1,
                padding="same",
            ),
            torch.nn.BatchNorm2d(num_features=12),
            torch.nn.ReLU(inplace=True),
            # [batch_size, 12, 32, 32]
            # torch.nn.Conv2d(
            #     in_channels=12,
            #     out_channels=24,
            #     kernel_size=5,
            #     stride=1,
            #     padding="same",
            # ),
            # torch.nn.BatchNorm2d(num_features=24),
            # torch.nn.ReLU(inplace=True),
            # # [batch_size, 24, 32, 32]
            # torch.nn.Conv2d(
            #     in_channels=24,
            #     out_channels=32,
            #     kernel_size=3,
            #     stride=1,
            #     padding="same",
            # ),
            # torch.nn.BatchNorm2d(num_features=32),
            # torch.nn.ReLU(inplace=True),
            # # [batch_size, 32, 32, 32]
            # torch.nn.Conv2d(
            #     in_channels=32,
            #     out_channels=64,
            #     kernel_size=5,
            #     stride=2,
            #     padding=0,
            # ),
            # torch.nn.BatchNorm2d(num_features=64),
            # torch.nn.ReLU(inplace=True),
            # # [batch_size, 64, 14, 14]
            # torch.nn.Conv2d(
            #     in_channels=64,
            #     out_channels=64,
            #     kernel_size=5,
            #     stride=2,
            #     padding=0,
            # ),
            # torch.nn.BatchNorm2d(num_features=64),
            # torch.nn.ReLU(inplace=True),
            # [batch_size, 64, 5, 5]
            torch.nn.Flatten(start_dim=1),
            # [batch_size, 1600]
            torch.nn.Linear(in_features=12 * 32 * 32, out_features=code_size),
            # torch.nn.Linear(in_features=64 * 5 * 5, out_features=code_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
        )
        # [batch_size, code_size]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor via the Encoder.

        Args:
            x (torch.Tensor):
                Input tensor of shape [batch_size, input_channels, height, width].

        Returns:
            output (torch.Tensor):
                Output tensor of shape [batch_size, code_size]
        """
        return self.net(x)


class Decoder(torch.nn.Module):
    def __init__(self, code_size: int, output_channels: int) -> None:
        """Decoder class.

        Implement a convolutional decoder that takes a 1D input tensor of size code_size
        and maps it into an output image of output_channels channels.

        Args:
            code_size (int):
                Size of the encoded representation.
            output_channels (int):
                Number of channels of output tensor/image.
        """
        super().__init__()

        # [batch_size, code_size]
        self.net = torch.nn.Sequential(
            # torch.nn.Linear(in_features=code_size, out_features=64 * 5 * 5),
            torch.nn.Linear(in_features=code_size, out_features=12 * 32 * 32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            # [batch_size, 1600]
            # torch.nn.Unflatten(dim=1, unflattened_size=(64, 5, 5)),
            torch.nn.Unflatten(dim=1, unflattened_size=(12, 32, 32)),
            # [batch_size, 64, 5, 5]
            # torch.nn.ConvTranspose2d(
            #     in_channels=64,
            #     out_channels=64,
            #     kernel_size=5,
            #     stride=2,
            #     padding=0,
            #     output_padding=1,
            # ),
            # torch.nn.BatchNorm2d(num_features=64),
            # torch.nn.ReLU(inplace=True),
            # # [batch_size, 64, 14, 14]
            # torch.nn.ConvTranspose2d(
            #     in_channels=64,
            #     out_channels=32,
            #     kernel_size=5,
            #     stride=2,
            #     padding=0,
            #     output_padding=1,
            # ),
            # torch.nn.BatchNorm2d(num_features=32),
            # torch.nn.ReLU(inplace=True),
            # # [batch_size, 32, 32, 32]
            # torch.nn.ConvTranspose2d(
            #     in_channels=32,
            #     out_channels=24,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            # ),
            # torch.nn.BatchNorm2d(num_features=24),
            # torch.nn.ReLU(inplace=True),
            # # [batch_size, 24, 32, 32]
            # torch.nn.ConvTranspose2d(
            #     in_channels=24,
            #     out_channels=12,
            #     kernel_size=5,
            #     stride=1,
            #     padding=2,
            # ),
            # torch.nn.BatchNorm2d(num_features=12),
            # torch.nn.ReLU(inplace=True),
            # [batch_size, 12, 32, 32]
            torch.nn.ConvTranspose2d(
                in_channels=12,
                out_channels=output_channels,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            torch.nn.BatchNorm2d(num_features=output_channels),
            torch.nn.Sigmoid(),
        )
        # [batch_size, output_channels, 32, 32]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor via the Decoder.

        Args:
            x (torch.Tensor):
                Input tensor of shape [batch_size, code_size]

        Returns:
            output (torch.Tensor):
                Output tensor of shape [batch_size, output_channels, height, width].
        """
        return self.net(x)


class FullyConnected(torch.nn.Module):
    def __init__(self, code_size: int, num_classes: int) -> None:
        """FullyConnected class.

        Implement a series of fully-connected layers that takes a 1D input tensor of
        size code_size and maps it into a 1D output tensor of size num_classes.

        Args:
            code_size (int):
                Size of the encoded representation.
            num_classes (int):
                Number of classes.
        """
        super().__init__()

        # [batch_size, code_size]
        self.net = torch.nn.Sequential(
            # torch.nn.Linear(in_features=code_size, out_features=int(code_size * 1.5)),
            torch.nn.Linear(in_features=code_size, out_features=num_classes),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            # torch.nn.Linear(
            #     in_features=int(code_size * 1.5), out_features=code_size * 2
            # ),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(),
            # torch.nn.Linear(
            #     in_features=code_size * 2, out_features=int(code_size * 0.5)
            # ),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(),
            # torch.nn.Linear(in_features=int(code_size * 0.5), out_features=num_classes),
        )
        # [batch_size, num_classes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor via the FullyConnected.

        Args:
            x (torch.Tensor):
                Input tensor of shape [batch_size, code_size]

        Returns:
            output (torch.Tensor):
                Output tensor of shape [batch_size, num_classes].
        """
        return self.net(x)


class BaseModel(torch.nn.Module, ABC):
    def __init__(
        self,
        code_size: int,
        device: str = "cuda:0",
    ) -> None:
        """BaseModel class.

        Implements the abstract architecture of the possible models.
        Subclasses must implement the _batch_step() and forward() functions.
        If a custom loss is provided for a particular model, then also the _loss function
        must be implemented.

        Args:
            code_size (int):
                Size of the encoded representation.
            device (str, optional):
                Running device, cpu or gpu. Default: "cuda:0".
        """
        super().__init__()

        # device setup
        self.device = (
            torch.device(device)
            if torch.cuda.is_available() and bool(re.findall("cuda:[\d]+$", device))
            else torch.device("cpu")
        )

        # model architecture
        self.code_size = code_size
        self.training_params = {"code_size": code_size}
        self.model = {}

    def _model_to_device(self):
        """Move the model to the defined device."""

        for module in self.model.values():
            module.to(self.device)
        # self.model = {
        #     module_name: module.to(self.device)
        #     for module_name, module in self.model.items()
        # }

    def save(self, path: str) -> None:
        """Save the model.

        The model parameters are saved to memory.

        Args:
            path (str):
                Path of the saved file, must have `.pth` extension
        """
        torch.save(
            {
                module_name: module  # .state_dict()
                for module_name, module in self.model.items()
            },
            path,
        )

    def load(self, path: str, **args) -> None:
        """Load the model.

        The model parameters are loaded from memory.

        Args:
            path (str)
                Path of the loaded file, must have `.pth` extension
            **args:
                (optional) parameters to pass to `torch.load_state_dict` function.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model = {
            module_name: checkpoint[module_name]
            # module_name: module.to(self.device).load_state_dict(
            #     checkpoint[module_name], **args
            # )
            for module_name, module in self.model.items()
        }

    def _model_name(self):

        if not hasattr(self, "training_params"):
            raise RuntimeError(f"compile function should be called before fit.")

        now = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
        code_size = self.code_size
        batch_size = self.training_params.get("batch_size", None)
        epochs = self.training_params.get("epochs", None)
        optimizer = self.training_params.get("optimizer", None)
        lr = self.training_params.get("lr", None)
        base_name = f"{code_size}-{batch_size}b-{epochs}e-{optimizer}-{lr}l-{now}"
        model_name = f"{type(self).__name__}-{base_name}"

        return model_name

    def _set_train_mode(self, mode: bool = True) -> torch.nn.Module:
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
        self.model = {
            module_name: module.train(mode=mode)
            for module_name, module in self.model.items()
        }

    def _is_training(self) -> bool:
        """Check if the model is in training mode.

        Returns:
            bool:
                Whether any of the submodels is in training mode.
        """

        return any([model.training for model in self.model.values()])

    def _loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the model.

        The given loss function (provided by the compile method) is used to compute the
        loss between the output and the expected target.

        Args:
            outputs (torch.Tensor):
                The output of the model, can have any dimension.
            targets (torch.Tensor):
                The target associated to the model output, can have any dimensions in
                accordance with the dimensions of the output.

        Returns:
            loss (float):
                Value of the loss function.
        """
        if self.loss_fn is None:
            raise RuntimeError(
                f"Loss function should be implemented in the calling class: {self.__class__}"
            )

        return self.loss_fn(outputs, targets)

    def _accuracy(
        self, logits: torch.Tensor, targets: torch.Tensor, reduction: str = "sum"
    ) -> Union[float, None]:
        """Compute the accuracy of the model.

        Args:
            logits (torch.Tensor):
                The output of the model, can have any dimension ().
            target (torch.Tensor):
                The target associated to the model output, can have any dimensions in
                accordance with the dimensions of the output.

        Returns:
            accuracy Union[float, None]:
                Accuracy of the model on the given outputs and targets. If accuracy is set
                to False in the compile method, then it returns None.
        """

        if not self.accuracy:
            return None

        if not isinstance(logits, list):
            logits = [logits]
        if not isinstance(targets, list):
            targets = [targets]

        outputs = torch.nn.functional.softmax(logits[0], dim=1)

        pred_labels = torch.argmax(outputs, dim=1)
        labels = torch.argmax(targets[0], dim=1)

        correct_decisions = torch.eq(labels, pred_labels).float()
        if reduction.lower() == "sum":
            accuracy = torch.sum(correct_decisions).item()
        elif reduction.lower() == "mean":
            accuracy = torch.mean(correct_decisions).item()
        else:
            raise ValueError(f"unrecognized reduction method: {reduction.lower()}")

        return accuracy

    @abstractmethod
    def forward(
        self, X: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the output of the network.

        The forward phase maps the input image into its encoded representation and the
        reconstructed image.

        Args:
            X (torch.Tensor):
                4D input tensor of the CNN [batch_size, 3, 32, 32].
            training (bool, optional):
                If True, the module is in training mode, otherwise it is in evalutation
                mode. Default: True.

        Returns:
            Outputs of the networks, or any intermediate representations.
        """
        pass

    @abstractmethod
    def _batch_step(
        self, X: torch.Tensor, Y: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def compile(
        self,
        optimizer: str,
        lr: float,
        loss: Union[str, Tuple[str, str], torch.nn.modules.loss._Loss, None],
        accuracy: bool = False,
    ) -> None:
        """Configures the model for training.

        Sets the optimizer, the learning rate, the loss function and the metrics.

        Args:
            optimizer (str):
                Optimizer. Possible choices are "Adam", "SGD", "RMSprop".
            lr (float):
                Learning rate for the optimizer.
            loss (Union[str, Tuple[str, str], torch.nn.modules.loss._Loss, None]):
                Loss function. If a tuple is provided, the first value is the name of the
                loss and the second one is the reduction method.
                Possible choices for the loss are "MSELoss", "CrossEntropyLoss", "L1Loss",
                "MultiMarginLoss". Possible choices for the reduction are "mean", "sum".
                It can also be a user defined loss of type torch.nn.modules.loss._Loss.
                If None, the _loss method must be override.
            accuracy (bool, optional):
                Whether to evaluate the model oaccuracy or not. Default: False
        """

        # set optimizer
        params_to_optimize = [
            {"params": filter(lambda p: p.requires_grad, model.parameters())}
            for _, model in self.model.items()
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
        elif isinstance(loss, torch.nn.modules.loss._Loss):
            loss_fn = loss
        elif loss is None:
            loss_fn = loss
        else:
            raise TypeError(
                f"loss should be of type str or tuple(str, str) or torch.nn.modules.loss._Loss. Instead {type(loss)}"
            )

        if isinstance(loss_fn, torch.nn.modules.loss._Loss) or loss_fn is None:
            self.loss_fn = loss_fn
        else:
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

        # set accuracy
        self.accuracy = accuracy

        self.training_params.update({"lr": lr, "optimizer": optimizer})

    def fit(
        self,
        training_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
        epochs: int = 1,
        model_checkpoint_folder: str = "vials-detection/models/",
    ) -> None:
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Args:
            training_loader (torch.utils.data.DataLoader):
                Training dataset DataLoader.
            validation_loader (torch.utils.data.DataLoader:
                Validation dataset DataLoader.
            epochs (int, optional):
                Number of training epochs. Default: 1.
            model_checkpoint_folder (str, optional):
                Path to the checkpoint folder. Default: "vials-detection/models/".
        """

        if not hasattr(self, "training_params"):
            raise RuntimeError(f"compile function should be called before fit.")

        self.training_params.update(
            {"epochs": epochs, "batch_size": training_loader.batch_size}
        )

        self.model_name = self._model_name()
        self.model_checkpoint_filepath = (
            f"{os.path.join(model_checkpoint_folder, self.model_name)}.pth"
        )

        if not os.path.exists(model_checkpoint_folder):
            os.makedirs(model_checkpoint_folder)

        # initialization
        self._set_train_mode(True)

        best_epoch = -1

        best_validation_loss = math.inf
        self.epochs_validation_loss_list = list()
        self.epochs_training_loss_list = list()

        best_validation_acc = -1.0 if self.accuracy else None
        self.epochs_validation_accs_list = list() if self.accuracy else None
        self.epochs_training_accs_list = list() if self.accuracy else None

        # training over epochs
        for e in range(0, epochs):

            print(f"Epoch {e + 1}/{epochs}")

            epoch_training_loss = 0.0
            epoch_training_acc = 0.0 if self.accuracy else None
            epoch_num_training_examples = 0

            # looping on batches
            for X, Y in tqdm.tqdm(training_loader):

                # if len(training_set) % batch_size != 0, then batch_size on last iteration is != from user batch_size
                batch_num_training_examples = X.shape[0]
                epoch_num_training_examples += batch_num_training_examples

                output, target = self._batch_step(X, Y, training=True)

                loss = self._loss(output, target)

                self.optimizer.zero_grad()  # put all gradients to zero before computing backward phase
                loss.backward()  # computing gradients (for parameters with requires_grad=True)
                self.optimizer.step()  # updating parameters according to optimizer

                # mini-batch evaluation
                with torch.no_grad():  # keeping off the autograd engine

                    if self.accuracy:
                        batch_training_acc = self._accuracy(
                            output, target, reduction="sum"
                        )
                        epoch_training_acc += batch_training_acc

                    batch_training_loss = loss.item()
                    epoch_training_loss += (
                        batch_training_loss * batch_num_training_examples
                    )

            # training set results
            epoch_training_loss /= epoch_num_training_examples
            self.epochs_training_loss_list.append(epoch_training_loss)

            if self.accuracy:
                epoch_training_acc /= epoch_num_training_examples
                self.epochs_training_accs_list.append(epoch_training_acc)

            # validation set evaluation
            epoch_validation_loss, epoch_validation_acc = self.evaluate(
                evaluation_loader=validation_loader
            )

            if self.accuracy:
                if epoch_validation_acc > best_validation_acc:
                    best_validation_acc = epoch_validation_acc
                    best_epoch = e + 1

                    self.save(path=self.model_checkpoint_filepath)

            else:
                if epoch_validation_loss < best_validation_loss:
                    best_validation_loss = epoch_validation_loss
                    best_epoch = e + 1

                    self.save(path=self.model_checkpoint_filepath)

            self.epochs_validation_loss_list.append(epoch_validation_loss)

            if self.accuracy:
                self.epochs_validation_accs_list.append(epoch_validation_acc)

            message = (
                f"train_loss: {epoch_training_loss:.4f} - "
                + (f"train_acc: {epoch_training_acc:.4f}" if self.accuracy else "")
                + f" - val_loss: {epoch_validation_loss:.4f} - "
                + (f"val_acc: {epoch_validation_acc:.4f}" if self.accuracy else "")
                + (" - BEST!" if best_epoch == e + 1 else "")
            )

            print(message)

        self._plot_curves()

    def evaluate(
        self, evaluation_loader: torch.utils.data.DataLoader, best_model: bool = False
    ) -> Tuple[float, Union[float, None]]:
        """Returns the loss value & metrics values for the model in evaluation mode.

        Args:
            evaluation_loader (torch.utils.data.DataLoader):
                Evaluation dataset DataLoader.
            best_model (bool, optional):
                Whether to use the best model. Default: False.

        Returns:
            loss (float):
                Loss of the network on evaluation dataset.
            acc (Union[float, None]):
                Accuracy of the network on evaluation dataset.
        """

        batch_outputs = [[]]
        batch_targets = [[]]

        if best_model:
            self.load(self.model_checkpoint_filepath)

        training_mode_originally_on = self._is_training()

        with torch.no_grad():  # keeping off autograd engine

            if training_mode_originally_on:
                self._set_train_mode(False)

            # mini-batch evaluation
            iteration = 0
            for X, Y in tqdm.tqdm(evaluation_loader):

                output, target = self._batch_step(X, Y, training=False)

                # append operation forced to be computed in cpu
                if (
                    iteration == 0
                    and isinstance(output, list)
                    and isinstance(target, list)
                ):
                    if len(output) != len(target):
                        raise RuntimeError(
                            f"outputs and targets have different length: {len(output)} vs {len(target)}"
                        )
                    batch_outputs = [[] for _ in range(len(output))]
                    batch_targets = [[] for _ in range(len(target))]

                if isinstance(output, list) and isinstance(target, list):
                    for i in range(len(output)):
                        batch_outputs[i].append(output[i].cpu())
                        batch_targets[i].append(target[i].cpu())
                else:
                    batch_outputs[0].append(output.cpu())
                    batch_targets[0].append(target.cpu())

                iteration += 1

            if isinstance(output, list) and isinstance(target, list):
                outputs = [
                    torch.cat(batch_outputs[i], dim=0) for i in range(len(output))
                ]
                targets = [
                    torch.cat(batch_targets[i], dim=0) for i in range(len(target))
                ]
            else:
                outputs = torch.cat(batch_outputs[0], dim=0)
                targets = torch.cat(batch_targets[0], dim=0)

            loss = self._loss(outputs, targets)
            acc = self._accuracy(outputs, targets, reduction="mean")

            if training_mode_originally_on:
                self._set_train_mode(True)

        return loss.item(), acc

    # def predict(self, test_loader: torch.utils.data.DataLoader):
    #     """Generates output predictions for the input samples.

    #     Args:
    #         test_loader (torch.utils.data.DataLoader):
    #             Test dataset DataLoader.
    #     """

    def _plot_curves(self) -> None:
        """Plots validation and training loss and accuracy curves."""

        # parameters retrieval
        code_size = self.training_params.get("code_size", None)
        batch_size = self.training_params.get("batch_size", None)
        epochs = self.training_params.get("epochs", None)
        optimizer = self.training_params.get("optimizer", None)
        lr = self.training_params.get("lr", None)

        # plotting
        x = list(range(1, epochs + 1))

        _, ax1 = plt.subplots()
        p1 = ax1.plot(x, self.epochs_training_loss_list, "-b", label="Training loss")
        p2 = ax1.plot(
            x, self.epochs_validation_loss_list, "--b", label="Validation loss"
        )
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.set_title(
            f"code_size={code_size}, batch_size={batch_size}, epochs={epochs}, optimizer={optimizer}, lr={lr}"
        )
        ax1.grid()

        if self.accuracy:
            ax2 = ax1.twinx()
            p3 = ax2.plot(x, self.epochs_training_accs_list, "-r", label="Training acc")
            p4 = ax2.plot(
                x, self.epochs_validation_accs_list, "--r", label="Validation acc"
            )
            ax2.set_ylabel("Accuracy")

        plots = p1 + p2 + p3 + p4 if self.accuracy else p1 + p2

        labs = [p.get_label() for p in plots]
        ax1.legend(plots, labs, loc=0)

        # save plot
        BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        FOLDER = os.path.join(BASEDIR, "results/")
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)
        FILEPATH = os.path.join(FOLDER, self.model_name)
        plt.savefig(f"{FILEPATH}.png", dpi=1000)


class AE(BaseModel):
    def __init__(self, code_size: int, device: str = "cpu") -> None:
        super().__init__(code_size=code_size, device=device)

        # model architecture
        encoder = Encoder(input_channels=3, code_size=code_size)
        decoder = Decoder(code_size=code_size, output_channels=3)
        self.model = {"encoder": encoder, "decoder": decoder}

        self._model_to_device()

    def forward(
        self, X: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        training_mode_originally_on = self._is_training()

        if training:
            self._set_train_mode(True)
        else:
            self._set_train_mode(False)

        encoded = self.model["encoder"](X)
        decoded = self.model["decoder"](encoded)

        if training_mode_originally_on:
            self._set_train_mode(True)

        return decoded, encoded

    def _batch_step(
        self, X: torch.Tensor, Y: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        X = X.to(self.device)
        Y = Y.to(self.device)

        # define target
        target = X

        output, _ = self.forward(X, training=training)

        return output, target


class CNN(BaseModel):
    def __init__(self, num_classes: int, code_size: int, device: str = "cpu") -> None:
        super().__init__(code_size, device=device)

        # model architecture
        self.num_classes = num_classes
        encoder = Encoder(input_channels=3, code_size=code_size)
        fully_connected = FullyConnected(code_size=code_size, num_classes=num_classes)
        self.model = {"encoder": encoder, "fully_connected": fully_connected}

        self._model_to_device()

    def forward(
        self, X: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        training_mode_originally_on = self._is_training()

        if training:
            self._set_train_mode(True)
        else:
            self._set_train_mode(False)

        encoded = self.model["encoder"](X)
        logits = self.model["fully_connected"](encoded)

        if training_mode_originally_on:
            self._set_train_mode(True)

        return logits, encoded

    def _batch_step(
        self, X: torch.Tensor, Y: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        X = X.to(self.device)
        Y = Y.to(self.device)

        # define target
        target = Y

        output, _ = self.forward(X, training=training)

        return output, target


class SAE(BaseModel):
    def __init__(self, num_classes: int, code_size: int, device: str = "cpu") -> None:
        super().__init__(code_size, device=device)

        # model architecture
        self.num_classes = num_classes
        encoder = Encoder(input_channels=3, code_size=code_size)
        decoder = Decoder(code_size=code_size, output_channels=3)
        fully_connected = FullyConnected(code_size=code_size, num_classes=num_classes)
        self.model = {
            "encoder": encoder,
            "decoder": decoder,
            "fully_connected": fully_connected,
        }

        self._model_to_device()

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        training_mode_originally_on = self._is_training()

        if training:
            self._set_train_mode(True)
        else:
            self._set_train_mode(False)

        encoded = self.model["encoder"](x)
        decoded = self.model["decoder"](encoded)
        logits = self.model["fully_connected"](encoded)

        if training_mode_originally_on:
            self._set_train_mode(True)

        return [logits, decoded], encoded

    def _batch_step(
        self, X, Y, training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        X = X.to(self.device)
        Y = Y.to(self.device)

        # define target
        targets = [Y, X]

        outputs, _ = self.forward(X, training=training)

        return outputs, targets

    def _loss(
        self,
        outputs: List[torch.Tensor],
        targets: List[torch.Tensor],
        supervision_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
    ) -> torch.Tensor:

        mse = torch.nn.MSELoss(reduction="mean")
        ce = torch.nn.CrossEntropyLoss(reduction="mean")

        logits = outputs[0]
        labels = targets[0]

        reconstructed_images = outputs[1]
        original_images = targets[1]

        return supervision_weight * ce(logits, labels) + reconstruction_weight * mse(
            reconstructed_images, original_images
        )
