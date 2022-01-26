import abc
import re
from typing import List, Tuple, Union

import torch
import torchmetrics
from torchmetrics import metric
import tqdm


class Encoder(torch.nn.Module):
    def __init__(
        self, n_channels: int, encoded_size: int, device: torch.device
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.encoded_size = encoded_size
        self.device = device

        # [n_channels, 90, 90]
        # [n_channels, 32, 32]
        self.conv_block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.n_channels,
                out_channels=12,
                # kernel_size=11,
                # stride=2,
                # padding=0,
                kernel_size=7,
                stride=2,
                padding=0,
            ),
            torch.nn.BatchNorm2d(num_features=12),
            torch.nn.ReLU(inplace=True),
        )
        # [12, 40, 40]
        # [12, 13, 13]
        self.conv_block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=12,
                out_channels=32,
                # kernel_size=7,
                # stride=2,
                # padding=0,
                kernel_size=5,
                stride=1,
                padding=0,
            ),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(inplace=True),
        )
        # [32, 17, 17]
        # [32, 9, 9]
        self.conv_block_3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                # kernel_size=5,
                # stride=2,
                # padding=0,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=True),
        )
        # [64, 7, 7]
        self.conv_block_4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=2,
                padding=0,
            ),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(inplace=True),
        )
        # [128, 2, 2]
        self.flatten = torch.nn.Flatten()
        # [128 * 2 * 2]
        self.linear_block_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=128 * 2 * 2, out_features=128 * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=128 * 2, out_features=self.encoded_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
        )
        # [encoded_size]

        self.to(self.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(input)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.flatten(x)
        encoded = self.linear_block_1(x)

        return encoded


class Decoder(torch.nn.Module):
    def __init__(
        self, n_channels: int, encoded_size: int, device: torch.device
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.encoded_size = encoded_size
        self.device = device

        # [encoded_size]
        self.linear_block_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.encoded_size, out_features=128 * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(in_features=128 * 2, out_features=128 * 2 * 2),
            torch.nn.ReLU(inplace=True),
        )
        # [128 * 2 * 2]
        self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(128, 2, 2))
        # [128, 2, 2]
        self.deconv_block_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=5,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=True),
        )
        # [64, 7, 7]
        self.deconv_block_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                # kernel_size=5,
                # stride=2,
                # padding=0,
                # output_padding=0,
                kernel_size=3,
                stride=1,
                padding=0,
                output_padding=0,
            ),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(inplace=True),
        )
        # [32, 17, 17]
        # [32, 9, 9]
        self.deconv_block_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=32,
                out_channels=12,
                # kernel_size=7,
                # stride=2,
                # padding=0,
                # output_padding=1,
                kernel_size=5,
                stride=1,
                padding=0,
                output_padding=0,
            ),
            torch.nn.BatchNorm2d(num_features=12),
            torch.nn.ReLU(inplace=True),
        )
        # [12, 40, 40]
        # [12, 13, 13]
        self.deconv_block_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=12,
                out_channels=self.n_channels,
                # kernel_size=11,
                # stride=2,
                # padding=0,
                # output_padding=1,
                kernel_size=7,
                stride=2,
                padding=0,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(num_features=self.n_channels),
        )
        # [n_channels, 90, 90]
        # [n_channels, 32, 32]

        self.to(device=self.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.linear_block_2(input)
        x = self.unflatten(x)
        x = self.deconv_block_1(x)
        x = self.deconv_block_2(x)
        x = self.deconv_block_3(x)
        decoded = self.deconv_block_4(x)

        return decoded


class OutputLayer(torch.nn.Module):
    def __init__(self, encoded_size: int, n_classes: int, device: torch.device) -> None:
        super().__init__()

        self.encoded_size = encoded_size
        self.n_classes = n_classes
        self.device = device

        # [encoded_size]
        self.linear_block_1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.encoded_size, out_features=128 * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(in_features=128 * 2, out_features=128 * 2 * 2),
            torch.nn.ReLU(inplace=True),
        )
        # [512]
        self.linear_block_2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=128 * 2 * 2, out_features=128 * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=128 * 2, out_features=self.n_classes),
        )
        # [n_classes]

        self.to(device=self.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.linear_block_1(input)
        output = self.linear_block_2(x)

        return output


class BaseModel(torch.nn.Module, abc.ABC):
    def __init__(
        self, n_channels: int, encoded_size: int, n_classes: int, device: str = "cuda:0"
    ) -> None:
        super().__init__()

        # device setup
        self.device = (
            torch.device(device)
            if torch.cuda.is_available() and bool(re.findall("cuda:[\d]+$", device))
            else torch.device("cpu")
        )

        # model architecture
        self.n_channels = n_channels
        self.encoded_size = encoded_size
        self.n_classes = n_classes

    def save(self, path: str) -> None:
        """Save the model.

        The model state_dict is saved to memory.

        Args:
            path (str):
                Path of the saved file, must have `.pth` extension
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the model.

        The model state_dict is loaded from memory.

        Args:
            path (str)
                Path of the loaded file, must have `.pth` extension
        """
        self.load_state_dict(torch.load(path))

    def _criterion(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
        return self.criterion(outputs, targets)

    def _metric(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Union[float, None]:
        """Compute the metrics of the model.

        Args:
            logits (torch.Tensor):
                The output of the model, can have any dimension ().
            target (torch.Tensor):
                The target associated to the model output, can have any dimensions in
                accordance with the dimensions of the output.

        Returns:
            global_accuracy (float):
                Accuracy of the model on the given outputs and targets, averaged over the
                classes.
            class_accuracy (torch.Tensor):
                Accuracy of the model on the given outputs and targets, for each class.

        Note:
            If a model is returning more than one outputs, then the first one should be
            the one over which to evaluate the metrics.
        """
        metrics_collection = torchmetrics.MetricCollection(
            {
                "mcc": torchmetrics.MatthewsCorrcoef(num_classes=self.n_classes),
                "accuracy": torchmetrics.Accuracy(average="micro"),
                "macro_f1score": torchmetrics.F1(
                    num_classes=self.n_classes, average="macro"
                ),
                "macro_precision": torchmetrics.Precision(
                    num_classes=self.n_classes, average="macro"
                ),
                "macro_recall": torchmetrics.Recall(
                    num_classes=self.n_classes, average="macro"
                ),
                "class_f1score": torchmetrics.F1(
                    num_classes=self.n_classes, average=None
                ),
                "class_precision": torchmetrics.Precision(
                    num_classes=self.n_classes, average=None
                ),
                "class_recall": torchmetrics.Recall(
                    num_classes=self.n_classes, average=None
                ),
            }
        )

        return metrics_collection(logits, targets)

    @abc.abstractmethod
    def forward(self, X: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Compute the output(s) of the network.

        Args:
            X (torch.Tensor):
                4D input tensor of the model [batch_size, 3, 32, 32].

        Returns:
            Outputs of the networks. If more than one outputs have to be returned than a
            list should be used.
        """
        pass

    @abc.abstractmethod
    def _set_targets(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def compile(
        self,
        criterion: torch.nn.modules.loss._Loss,
        optimizer: str,
        learning_rate: float,
    ) -> None:
        """Configure the model for training.

        Sets the optimizer, the learning rate, the loss function and the metrics.

        Args:
            criterion (torch.nn.modules.loss._Loss):
                Loss function. It can be any of the PyTorch loss functions or a user
                defined loss of type torch.nn.modules.loss._Loss.
            optimizer (str):
                Optimizer. Possible choices are "Adam", "SGD".
            learning_rate (float):
                Learning rate for the optimizer.
        """

        # set optimizer
        params_to_optimize = filter(lambda p: p.requires_grad, self.parameters())
        if isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                self.optimizer = torch.optim.Adam(
                    params=params_to_optimize,
                    lr=learning_rate,
                )
            elif optimizer.lower() == "sgd":
                self.optimizer = torch.optim.SGD(
                    params=params_to_optimize,
                    lr=learning_rate,
                )
            else:
                print(f"unrecognized optimizer: {optimizer}. Set to Adam.")
                self.optimizer = torch.optim.Adam(
                    params=params_to_optimize,
                    lr=learning_rate,
                )
        else:
            raise ValueError(
                f"optimizer should be of type str. Instead {type(optimizer)}"
            )

        # set criterion
        if isinstance(criterion, torch.nn.modules.loss._Loss):
            self.criterion = criterion
        else:
            raise ValueError(f"unrecognized loss: {criterion}.")

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        epochs: int = 1,
        model_path: str = "vials-detection/models/new.pth",
    ) -> None:
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Args:
            train_loader (torch.utils.data.DataLoader):
                Training dataset DataLoader.
            valid_loader (torch.utils.data.DataLoader:
                Validation dataset DataLoader.
            epochs (int, optional):
                Number of training epochs. Default: 1.
            model_path (str, optional):
                Path to the model for saving. Default: "vials-detection/models/new.pth".
        """

        self.metric = torchmetrics.MetricCollection(
            {
                "mcc": torchmetrics.MatthewsCorrcoef(num_classes=self.n_classes),
                "accuracy": torchmetrics.Accuracy(average="micro"),
                "macro_f1score": torchmetrics.F1(
                    num_classes=self.n_classes, average="macro"
                ),
                "macro_precision": torchmetrics.Precision(
                    num_classes=self.n_classes, average="macro"
                ),
                "macro_recall": torchmetrics.Recall(
                    num_classes=self.n_classes, average="macro"
                ),
                "class_f1score": torchmetrics.F1(
                    num_classes=self.n_classes, average=None
                ),
                "class_precision": torchmetrics.Precision(
                    num_classes=self.n_classes, average=None
                ),
                "class_recall": torchmetrics.Recall(
                    num_classes=self.n_classes, average=None
                ),
            }
        ).to(self.device)

        best_epoch = -1

        self.epochs_valid_loss_list = list()
        self.epochs_train_loss_list = list()

        best_valid_mcc = -1.0
        # self.epochs_valid_mcc_list = list()
        # self.epochs_valid_accuracy_list = list()
        # self.epochs_valid_f1score_list = list()
        # self.epochs_valid_precision_list = list()
        # self.epochs_valid_recall_list = list()

        # self.epochs_train_mcc_list = list()
        # self.epochs_train_accuracy_list = list()
        # self.epochs_train_f1score_list = list()
        # self.epochs_train_precision_list = list()
        # self.epochs_train_recall_list = list()

        # training over epochs
        for e in range(0, epochs):

            print(f"Epoch {e + 1}/{epochs}")

            # TRAINING - start
            # set the network in training mode
            self.train()

            if self.multi_output:
                epoch_train_outputs_decoder = []
                epoch_train_targets_decoder = []
            epoch_train_logits = []
            epoch_train_targets = []

            # TRAINING BATCHES LOOP - start
            for X, Y in tqdm.tqdm(train_loader):
                # X: [batch_size, 3, 32, 32], Y: [batch_size,]
                X = X.to(self.device)
                Y = Y.to(self.device)

                targets = self._set_targets(X, Y)

                # TRAINING STEP - start
                outputs = self.forward(X)
                loss = self._criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # TRAINING STEP - end

                # TRAINING BATCHES EVALUATION - start
                with torch.no_grad():
                    # stacking predictions and targets for metrics and loss
                    if self.multi_output:
                        # handle case of SAE model (logits, reconstructed_images)
                        if not isinstance(outputs, list) or not isinstance(
                            targets, list
                        ):
                            raise RuntimeError(
                                f"outputs and targets should be lists. Instead {type(outputs)} and {type(targets)}"
                            )
                        epoch_train_outputs_decoder.append(outputs[1].cpu())
                        epoch_train_targets_decoder.append(outputs[1].cpu())
                        # for_metrics_outputs = outputs[0]
                        # for_metrics_targets = targets[0]
                        # for_loss_outputs = torch.cat[outputs[0]
                        # for_loss_targets = targets[0]

                    # metrics = self.metric(outputs, outputs)
                    # loss =
                    epoch_train_logits.append(outputs.cpu())
                    epoch_train_targets.append(targets.cpu())
                # TRAINING BATCHES EVALUATION - end

            # TRAINING BATCHES LOOP - end

            # TRAINING EPOCH EVALUATION - start
            # computing epoch loss
            epoch_train_loss = self._criterion(
                [
                    torch.cat(epoch_train_logits, dim=0),
                    torch.cat(epoch_train_outputs_decoder, dim=0),
                ]
                if self.multi_output
                else torch.cat(epoch_train_logits, dim=0),
                [
                    torch.cat(epoch_train_targets, dim=0),
                    torch.cat(epoch_train_targets_decoder, dim=0),
                ]
                if self.multi_output
                else torch.cat(epoch_train_targets, dim=0),
            )
            self.epochs_train_loss_list.append(epoch_train_loss.item())

            # computing epoch metric
            (
                epoch_train_mcc,
                epoch_train_accuracy,
                epoch_train_f1score,
                epoch_train_precision,
                epoch_train_recall,
                epoch_train_class_f1score,
                epoch_train_class_precision,
                epoch_train_class_recall,
            ) = self._metric(
                torch.cat(epoch_train_logits, dim=0),
                torch.cat(epoch_train_targets, dim=0),
            )
            self.epochs_train_mcc_list.append(epoch_train_mcc.item())
            self.epochs_train_accuracy_list.append(epoch_train_accuracy.item())
            self.epochs_train_f1score_list.append(epoch_train_f1score.item())
            self.epochs_train_precision_list.append(epoch_train_precision.item())
            self.epochs_train_recall_list.append(epoch_train_recall.item())
            # self.epochs_train_precision_list.append(
            #     [a.item() for a in epoch_train_precision]
            # )
            # TRAINING EPOCH EVALUATION - end

            # TRAINING - end

            # VALIDATION - start

            # VALIDATION EPOCH EVALUATION - start
            (
                epoch_valid_loss,
                epoch_valid_mcc,
                epoch_valid_accuracy,
                epoch_valid_f1score,
                epoch_valid_precision,
                epoch_valid_recall,
            ) = self.evaluate(eval_loader=valid_loader)
            self.epochs_valid_loss_list.append(epoch_valid_loss.item())
            self.epochs_valid_mcc_list.append(epoch_valid_mcc.item())
            self.epochs_valid_accuracy_list.append(epoch_valid_accuracy.item())
            self.epochs_valid_f1score_list.append(epoch_valid_f1score.item())
            self.epochs_valid_precision_list.append(epoch_valid_precision.item())
            self.epochs_valid_recall_list.append(epoch_valid_recall.item())
            # self.epochs_valid_precision_list.append(
            #     [a.item() for a in epoch_valid_precision]
            # )
            # VALIDATION EPOCH EVALUATION - end

            # END EPOCHS CHECKS - start
            if epoch_valid_accuracy > best_valid_mcc:
                best_valid_mcc = epoch_valid_accuracy
                best_epoch = e + 1
                torch.save(self.state_dict(), model_path)

            message = (
                f"TRAIN | loss: {epoch_train_loss:.4f}"
                + f" - mcc: {epoch_train_mcc:.4f}"
                + f" - accuracy: {epoch_train_accuracy:.4f}"
                + f" - f1score: {epoch_train_f1score:.4f}"
                + f" - precision: {epoch_train_precision:.4f}"
                + f" - recall: {epoch_train_recall:.4f}"
                + f" \nVALID | loss: {epoch_valid_loss:.4f}"
                + f" - mcc: {epoch_valid_mcc:.4f}"
                + f" - accuracy: {epoch_valid_accuracy:.4f}"
                + f" - f1score: {epoch_valid_f1score:.4f}"
                + f" - precision: {epoch_valid_precision:.4f}"
                + f" - recall: {epoch_valid_recall:.4f}"
                # + " ".join([f"{a.item():.4f}" for a in epoch_valid_precision])
                # + "]"
                + (" - BEST!" if best_epoch == e + 1 else "")
            )

            print(message)

    def evaluate(
        self,
        eval_loader: torch.utils.data.DataLoader,
        model_path: Union[str, None] = None,
    ) -> Tuple[float, Union[float, None]]:
        """Returns the loss value & metrics values for the model in evaluation mode.

        Args:
            evaluation_loader (torch.utils.data.DataLoader):
                Evaluation dataset DataLoader.
            model_path (Union[str, None], optional):
                The path to the model to be used for evaluation. Default: None.

        Returns:
            loss (float):
                Loss of the network on evaluation dataset.
            global_accuracy (Union[float, None]):
                Accuracy of the network on evaluation dataset.
        """

        if model_path is not None:
            self.load(model_path)

        training_mode_originally_on = self.training

        with torch.no_grad():  # keeping off autograd engine

            if training_mode_originally_on:
                self.eval()

            if self.multi_output:
                batches_outputs_dec = []
                batches_targets_dec = []
            batches_logits = []
            batches_targets = []

            # BATCHES LOOP - start
            for X, Y in tqdm.tqdm(eval_loader):
                # X: [batch_size, 3, 32, 32], Y: [batch_size, n_classes]
                X = X.to(self.device)
                Y = Y.to(self.device)

                targets = self._set_targets(X, Y)

                # BATCH STEP - start
                outputs = self.forward(X)
                # BATCH STEP - end

                # BATCHES EVALUATION - start
                # stacking predictions and targets for metrics and loss
                if self.multi_output:
                    # handle case of SAE model (logits, reconstructed_images)
                    if not isinstance(outputs, list) or not isinstance(targets, list):
                        raise RuntimeError(
                            f"outputs and targets should be lists. Instead {type(outputs)} and {type(targets)}"
                        )
                    batches_outputs_dec.append(outputs[1].cpu())
                    batches_targets_dec.append(outputs[1].cpu())
                    outputs = outputs[0]
                    targets = targets[0]
                batches_logits.append(outputs.cpu())
                batches_targets.append(targets.cpu())
                # BATCHES EVALUATION - end

            # LOSS and METRICS EVALUATION - start
            # computing validation loss
            loss = self._criterion(
                [
                    torch.cat(batches_logits, dim=0),
                    torch.cat(batches_outputs_dec, dim=0),
                ]
                if self.multi_output
                else torch.cat(batches_logits, dim=0),
                [
                    torch.cat(batches_targets, dim=0),
                    torch.cat(batches_targets_dec, dim=0),
                ]
                if self.multi_output
                else torch.cat(batches_targets, dim=0),
            )

            # computing validation metric
            (
                mcc,
                accuracy,
                f1score,
                precision,
                recall,
                class_f1score,
                class_precision,
                class_recall,
            ) = self._metric(
                torch.cat(batches_logits, dim=0),
                torch.cat(batches_targets, dim=0),
            )
            # LOSS and METRICS EVALUATION - end

            if training_mode_originally_on:
                self.train()

        return loss, mcc, accuracy, f1score, precision, recall


class SupervisedAutoencoder(BaseModel):
    def __init__(
        self, n_channels: int, encoded_size: int, n_classes: int, device: str = "cuda:0"
    ) -> None:
        super().__init__(
            n_channels=n_channels,
            encoded_size=encoded_size,
            n_classes=n_classes,
            device=device,
        )

        self.multi_output = True

        # [n_channels, 90, 90] = 24300
        self.encoder = Encoder(
            n_channels=self.n_channels,
            encoded_size=self.encoded_size,
            device=self.device,
        )
        # [encoded_size]

        # [encoded_size]
        self.decoder = Decoder(
            n_channels=self.n_channels,
            encoded_size=self.encoded_size,
            device=self.device,
        )
        # [n_channels, 90, 90]

        # [encoded_size]
        self.output_layer = OutputLayer(
            encoded_size=self.encoded_size,
            n_classes=self.n_classes,
            device=self.device,
        )
        # [n_classes]

        self.to(device=self.device)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.to(self.device)
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        output = self.output_layer(encoded)

        return [output, decoded]  # , encoded

    def _set_targets(
        self, X: torch.Tensor, Y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return [Y, X]


class ConvolutionalNetwork(BaseModel):
    def __init__(
        self, n_channels: int, encoded_size: int, n_classes: int, device: str = "cuda:0"
    ) -> None:
        super().__init__(
            n_channels=n_channels,
            encoded_size=encoded_size,
            n_classes=n_classes,
            device=device,
        )

        self.multi_output = False

        # [n_channels, 90, 90] = 24300
        self.encoder = Encoder(
            n_channels=self.n_channels,
            encoded_size=self.encoded_size,
            device=self.device,
        )
        # [encoded_size]

        # [encoded_size]
        self.output_layer = OutputLayer(
            encoded_size=self.encoded_size,
            n_classes=self.n_classes,
            device=self.device,
        )
        # [n_channels, 90, 90]

        self.to(device=self.device)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.device)
        encoded = self.encoder(input)
        output = self.output_layer(encoded)

        return output  # , encoded

    def _set_targets(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return Y


# def fit(
#     self,
#     train_loader,
#     valid_loader,
#     epochs,
#     classes,
#     learning_rate,
#     model_path,
# ):
#     best_epoch = -1

#     epochs_val_loss_list = list()
#     epochs_train_loss_list = list()

#     best_val_acc = -1.0
#     epochs_val_global_acc_list = list()
#     epochs_val_class_acc_list = list()
#     epochs_train_global_acc_list = list()
#     epochs_train_class_acc_list = list()

#     criterion = torch.nn.CrossEntropyLoss(reduction="sum")
#     optimizer = torch.optim.Adam(
#         params=filter(lambda p: p.requires_grad, self.parameters()),
#         lr=learning_rate,
#     )
#     global_acc = torchmetrics.Accuracy(num_classes=len(classes), average="weighted")
#     class_acc = torchmetrics.Accuracy(num_classes=len(classes), average="none")

#     for epoch in range(epochs):

#         # TRAINING - start

#         self.train()

#         epoch_train_loss = 0.0
#         epoch_train_global_acc = 0.0
#         epoch_num_train_examples = 0
#         epoch_train_outputs_for_metric = []
#         epoch_train_targets_for_metric = []

#         # TRAINING BATCHES LOOP - start
#         for X, Y in tqdm.tqdm(train_loader, 0):
#             # X: [batch_size, 3, 32, 32], Y: [batch_size,]
#             X = X.to(self.device)
#             Y = Y.to(self.device)

#             # targets definition
#             targets = Y

#             # needed to handle last iteration (if len(train_set) % batch_size != 0)
#             batch_num_train_examples = X.shape[0]
#             epoch_num_train_examples += batch_num_train_examples

#             # TRAINING STEP - start
#             outputs = self.forward(X)
#             loss = criterion(outputs, targets)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             # TRAINING STEP - end

#             # TRAINING BATCHES EVALUATION - start
#             with torch.no_grad():
#                 # accumulating batch loss
#                 epoch_train_loss += loss.item()
#                 # stacking predictions and targets for metric
#                 pred = outputs
#                 target = targets
#                 epoch_train_outputs_for_metric.append(pred.cpu())
#                 epoch_train_targets_for_metric.append(target.cpu())
#             # TRAINING BATCHES EVALUATION - end

#         # TRAINING BATCHES LOOP - end

#         # TRAINING - end

#         # VALIDATION - start
#         with torch.no_grad():

#             train_mode_originally_on = self.training
#             if train_mode_originally_on:
#                 self.eval()

#             epoch_val_outputs_for_metric = []
#             epoch_val_targets_for_metric = []

#             epoch_val_loss = 0.0
#             epoch_val_global_acc = 0.0
#             epoch_num_val_examples = 0

#             # VALIDATION BATCHES LOOP - start
#             for X, Y in tqdm.tqdm(valid_loader):
#                 # X: [batch_size, 3, 32, 32], Y: [batch_size, n_classes]
#                 X = X.to(self.device)
#                 Y = Y.to(self.device)

#                 # needed to handle last iteration (if len(val_set) % batch_size != 0)
#                 batch_num_val_examples = X.shape[0]
#                 epoch_num_val_examples += batch_num_val_examples

#                 # targets definition
#                 targets = Y

#                 # VALIDATION STEP - start
#                 outputs = self.forward(X)
#                 loss = criterion(outputs, targets)
#                 # VALIDATION STEP - end

#                 # VALIDATION BATCHES EVALUATION - start
#                 # accumulating batch loss
#                 epoch_val_loss += loss.item()
#                 # stacking predictions and targets for metrics
#                 pred = outputs
#                 target = targets
#                 epoch_val_outputs_for_metric.append(pred.cpu())
#                 epoch_val_targets_for_metric.append(target.cpu())
#                 # TRAINING BATCHES EVALUATION - end

#             # VALIDATION BATCHES LOOP - end

#             if train_mode_originally_on:
#                 self.train()

#         # VALIDATION - end

#         # TRAINING EPOCH EVALUATION - start

#         # computing epoch loss
#         epoch_train_loss /= epoch_num_train_examples
#         epochs_train_loss_list.append(epoch_train_loss)

#         # computing epoch metric
#         epoch_train_global_acc = global_acc(
#             torch.cat(epoch_train_outputs_for_metric, dim=0),
#             torch.cat(epoch_train_targets_for_metric, dim=0),
#         )
#         epochs_train_global_acc_list.append(epoch_train_global_acc)
#         epoch_train_class_acc = class_acc(
#             torch.cat(epoch_train_outputs_for_metric, dim=0),
#             torch.cat(epoch_train_targets_for_metric, dim=0),
#         )
#         epochs_train_class_acc_list.append(epoch_train_class_acc)
#         # TRAINING EPOCH EVALUATION - end

#         # VALIDATION EPOCH EVALUATION - start
#         # computing epoch loss
#         epoch_val_loss /= epoch_num_val_examples
#         epochs_val_loss_list.append(epoch_val_loss)

#         # computing epoch metric
#         epoch_val_global_acc = global_acc(
#             torch.cat(epoch_val_outputs_for_metric, dim=0),
#             torch.cat(epoch_val_targets_for_metric, dim=0),
#         )
#         epochs_val_global_acc_list.append(epoch_val_global_acc)
#         epoch_val_class_acc = class_acc(
#             torch.cat(epoch_val_outputs_for_metric, dim=0),
#             torch.cat(epoch_val_targets_for_metric, dim=0),
#         )
#         epochs_val_class_acc_list.append(epoch_val_class_acc)

#         # VALIDATION EPOCH EVALUATION - end

#         # END EPOCHS CHECKS - start
#         if epoch_val_global_acc > best_val_acc:
#             best_val_acc = epoch_val_global_acc
#             best_epoch = epoch + 1
#             torch.save(self.state_dict(), model_path)

#         message = (
#             f"Epoch: {epoch + 1}"
#             + f"\nTRAIN | loss: {epoch_train_loss:.4f}"
#             + f" - global_acc: {epoch_train_global_acc:.4f}"
#             + f" - class_acc: ["
#             + " ".join([f"{a.item():.4f}" for a in epoch_train_class_acc])
#             + "]"
#             + f" \nVALID | loss: {epoch_val_loss:.4f}"
#             + f" - global_acc: {epoch_val_global_acc:.4f}"
#             + f" - class_acc: ["
#             + " ".join([f"{a.item():.4f}" for a in epoch_val_class_acc])
#             + "]"
#             + (" - BEST!" if best_epoch == epoch + 1 else "")
#         )

#         print(message)
