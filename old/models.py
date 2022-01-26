import collections
import torch
import re
import torchmetrics
import tqdm
import math

from . import utils


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


class SupervisedAutoencoder(torch.nn.Module):
    def __init__(
        self, n_channels: int, encoded_size: int, n_classes: int, device: str = "cuda:0"
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.encoded_size = encoded_size
        self.n_classes = n_classes
        self.device = (
            torch.device(device)
            if torch.cuda.is_available() and bool(re.findall("cuda:[\d]+$", device))
            else torch.device("cpu")
        )

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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self.device)
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        output = self.output_layer(encoded)

        return [output, decoded]  # , encoded

    def fit(
        self,
        train_loader,
        valid_loader,
        epochs,
        classes,
        learning_rate,
        model_path,
    ):
        best_epoch = -1

        epochs_val_loss_list = list()
        epochs_train_loss_list = list()

        best_val_acc = -1.0
        epochs_val_global_acc_list = list()
        epochs_val_class_acc_list = list()
        epochs_train_global_acc_list = list()
        epochs_train_class_acc_list = list()

        criterion = utils.SAELoss(
            reduction="sum", supervision_weight=1.0, reconstruction_weight=1.0
        )
        optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.parameters()),
            lr=learning_rate,
        )
        global_acc = torchmetrics.Accuracy(num_classes=len(classes), average="weighted")
        class_acc = torchmetrics.Accuracy(num_classes=len(classes), average="none")

        for epoch in range(epochs):

            # TRAINING - start

            self.train()

            epoch_train_loss = 0.0
            epoch_num_train_examples = 0
            epoch_train_outputs_for_metric = []
            epoch_train_targets_for_metric = []

            # TRAINING BATCHES LOOP - start
            for X, Y in tqdm.tqdm(train_loader, 0):
                # X: [batch_size, 3, 32, 32], Y: [batch_size,]
                X = X.to(self.device)
                Y = Y.to(self.device)

                # targets definition
                targets = [Y, X]

                # needed to handle last iteration (if len(train_set) % batch_size != 0)
                batch_num_train_examples = X.shape[0]
                epoch_num_train_examples += batch_num_train_examples

                # TRAINING STEP - start
                outputs = self.forward(X)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # TRAINING STEP - end

                # TRAINING BATCHES EVALUATION - start
                with torch.no_grad():
                    # accumulating batch loss
                    epoch_train_loss += loss.item()
                    # stacking predictions and targets for metric
                    pred = outputs[0]
                    target = targets[0]
                    epoch_train_outputs_for_metric.append(pred.cpu())
                    epoch_train_targets_for_metric.append(target.cpu())
                # TRAINING BATCHES EVALUATION - end

            # TRAINING BATCHES LOOP - end

            # TRAINING EPOCH EVALUATION - start
            # computing epoch loss
            epoch_train_loss /= epoch_num_train_examples
            epochs_train_loss_list.append(epoch_train_loss)
            # computing epoch metric
            epoch_train_global_acc = global_acc(
                torch.cat(epoch_train_outputs_for_metric, dim=0),
                torch.cat(epoch_train_targets_for_metric, dim=0),
            )
            epochs_train_global_acc_list.append(epoch_train_global_acc)
            epoch_train_class_acc = class_acc(
                torch.cat(epoch_train_outputs_for_metric, dim=0),
                torch.cat(epoch_train_targets_for_metric, dim=0),
            )
            epochs_train_class_acc_list.append(epoch_train_class_acc)
            # TRAINING EPOCH EVALUATION - end

            # TRAINING - end

            # VALIDATION - start
            with torch.no_grad():

                train_mode_originally_on = self.training
                if train_mode_originally_on:
                    self.eval()

                epoch_val_outputs_for_metric = []
                epoch_val_targets_for_metric = []

                epoch_val_loss = 0.0
                epoch_num_val_examples = 0

                # VALIDATION BATCHES LOOP - start
                for X, Y in tqdm.tqdm(valid_loader):
                    # X: [batch_size, 3, 32, 32], Y: [batch_size, n_classes]
                    X = X.to(self.device)
                    Y = Y.to(self.device)

                    # needed to handle last iteration (if len(val_set) % batch_size != 0)
                    batch_num_val_examples = X.shape[0]
                    epoch_num_val_examples += batch_num_val_examples

                    # targets definition
                    targets = [Y, X]

                    # VALIDATION STEP - start
                    outputs = self.forward(X)
                    loss = criterion(outputs, targets)
                    # VALIDATION STEP - end

                    # VALIDATION BATCHES EVALUATION - start
                    # accumulating batch loss
                    epoch_val_loss += loss.item()
                    # stacking predictions and targets for metrics
                    pred = outputs[0]
                    target = targets[0]
                    epoch_val_outputs_for_metric.append(pred.cpu())
                    epoch_val_targets_for_metric.append(target.cpu())
                    # VALIDATION BATCHES EVALUATION - end

                # VALIDATION BATCHES LOOP - end

                if train_mode_originally_on:
                    self.train()

            # VALIDATION - end

            # VALIDATION EPOCH EVALUATION - start

            # computing epoch loss
            epoch_val_loss /= epoch_num_val_examples
            epochs_val_loss_list.append(epoch_val_loss)

            # computing epoch metric
            epoch_val_global_acc = global_acc(
                torch.cat(epoch_val_outputs_for_metric, dim=0),
                torch.cat(epoch_val_targets_for_metric, dim=0),
            )
            epochs_val_global_acc_list.append(epoch_val_global_acc)
            epoch_val_class_acc = class_acc(
                torch.cat(epoch_val_outputs_for_metric, dim=0),
                torch.cat(epoch_val_targets_for_metric, dim=0),
            )
            epochs_val_class_acc_list.append(epoch_val_class_acc)

            # VALIDATION EPOCH EVALUATION - end

            # END EPOCHS CHECKS - start
            if epoch_val_global_acc > best_val_acc:
                best_val_acc = epoch_val_global_acc
                best_epoch = epoch + 1
                torch.save(self.state_dict(), model_path)

            message = (
                f"Epoch: {epoch + 1}"
                + f"\nTRAIN | loss: {epoch_train_loss:.4f}"
                + f" - global_acc: {epoch_train_global_acc:.4f}"
                + f" - class_acc: ["
                + " ".join([f"{a.item():.4f}" for a in epoch_train_class_acc])
                + "]"
                + f" \nVALID | loss: {epoch_val_loss:.4f}"
                + f" - global_acc: {epoch_val_global_acc:.4f}"
                + f" - class_acc: ["
                + " ".join([f"{a.item():.4f}" for a in epoch_val_class_acc])
                + "]"
                + (" - BEST!" if best_epoch == epoch + 1 else "")
            )

            print(message)


class ConvolutionalNetwork(torch.nn.Module):
    def __init__(
        self, n_channels: int, encoded_size: int, n_classes: int, device: str = "cuda:0"
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.encoded_size = encoded_size
        self.n_classes = n_classes
        self.device = (
            torch.device(device)
            if torch.cuda.is_available() and bool(re.findall("cuda:[\d]+$", device))
            else torch.device("cpu")
        )

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

    def fit(
        self,
        train_loader,
        valid_loader,
        epochs,
        classes,
        learning_rate,
        model_path,
    ):
        best_epoch = -1

        epochs_val_loss_list = list()
        epochs_train_loss_list = list()

        best_val_acc = -1.0
        epochs_val_global_acc_list = list()
        epochs_val_class_acc_list = list()
        epochs_train_global_acc_list = list()
        epochs_train_class_acc_list = list()

        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.parameters()),
            lr=learning_rate,
        )
        global_acc = torchmetrics.Accuracy(num_classes=len(classes), average="weighted")
        class_acc = torchmetrics.Accuracy(num_classes=len(classes), average="none")

        for epoch in range(epochs):

            # TRAINING - start

            self.train()

            epoch_train_loss = 0.0
            epoch_train_global_acc = 0.0
            epoch_num_train_examples = 0
            epoch_train_outputs_for_metric = []
            epoch_train_targets_for_metric = []

            # TRAINING BATCHES LOOP - start
            for X, Y in tqdm.tqdm(train_loader, 0):
                # X: [batch_size, 3, 32, 32], Y: [batch_size,]
                X = X.to(self.device)
                Y = Y.to(self.device)

                # targets definition
                targets = Y

                # needed to handle last iteration (if len(train_set) % batch_size != 0)
                batch_num_train_examples = X.shape[0]
                epoch_num_train_examples += batch_num_train_examples

                # TRAINING STEP - start
                outputs = self.forward(X)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # TRAINING STEP - end

                # TRAINING BATCHES EVALUATION - start
                with torch.no_grad():
                    # accumulating batch loss
                    epoch_train_loss += loss.item()
                    # stacking predictions and targets for metric
                    pred = outputs
                    target = targets
                    epoch_train_outputs_for_metric.append(pred.cpu())
                    epoch_train_targets_for_metric.append(target.cpu())
                # TRAINING BATCHES EVALUATION - end

            # TRAINING BATCHES LOOP - end

            # TRAINING - end

            # VALIDATION - start
            with torch.no_grad():

                train_mode_originally_on = self.training
                if train_mode_originally_on:
                    self.eval()

                epoch_val_outputs_for_metric = []
                epoch_val_targets_for_metric = []

                epoch_val_loss = 0.0
                epoch_val_global_acc = 0.0
                epoch_num_val_examples = 0

                # VALIDATION BATCHES LOOP - start
                for X, Y in tqdm.tqdm(valid_loader):
                    # X: [batch_size, 3, 32, 32], Y: [batch_size, n_classes]
                    X = X.to(self.device)
                    Y = Y.to(self.device)

                    # needed to handle last iteration (if len(val_set) % batch_size != 0)
                    batch_num_val_examples = X.shape[0]
                    epoch_num_val_examples += batch_num_val_examples

                    # targets definition
                    targets = Y

                    # VALIDATION STEP - start
                    outputs = self.forward(X)
                    loss = criterion(outputs, targets)
                    # VALIDATION STEP - end

                    # VALIDATION BATCHES EVALUATION - start
                    # accumulating batch loss
                    epoch_val_loss += loss.item()
                    # stacking predictions and targets for metrics
                    pred = outputs
                    target = targets
                    epoch_val_outputs_for_metric.append(pred.cpu())
                    epoch_val_targets_for_metric.append(target.cpu())
                    # TRAINING BATCHES EVALUATION - end

                # VALIDATION BATCHES LOOP - end

                if train_mode_originally_on:
                    self.train()

            # VALIDATION - end

            # TRAINING EPOCH EVALUATION - start

            # computing epoch loss
            epoch_train_loss /= epoch_num_train_examples
            epochs_train_loss_list.append(epoch_train_loss)

            # computing epoch metric
            epoch_train_global_acc = global_acc(
                torch.cat(epoch_train_outputs_for_metric, dim=0),
                torch.cat(epoch_train_targets_for_metric, dim=0),
            )
            epochs_train_global_acc_list.append(epoch_train_global_acc)
            epoch_train_class_acc = class_acc(
                torch.cat(epoch_train_outputs_for_metric, dim=0),
                torch.cat(epoch_train_targets_for_metric, dim=0),
            )
            epochs_train_class_acc_list.append(epoch_train_class_acc)
            # TRAINING EPOCH EVALUATION - end

            # VALIDATION EPOCH EVALUATION - start
            # computing epoch loss
            epoch_val_loss /= epoch_num_val_examples
            epochs_val_loss_list.append(epoch_val_loss)

            # computing epoch metric
            epoch_val_global_acc = global_acc(
                torch.cat(epoch_val_outputs_for_metric, dim=0),
                torch.cat(epoch_val_targets_for_metric, dim=0),
            )
            epochs_val_global_acc_list.append(epoch_val_global_acc)
            epoch_val_class_acc = class_acc(
                torch.cat(epoch_val_outputs_for_metric, dim=0),
                torch.cat(epoch_val_targets_for_metric, dim=0),
            )
            epochs_val_class_acc_list.append(epoch_val_class_acc)

            # VALIDATION EPOCH EVALUATION - end

            # END EPOCHS CHECKS - start
            if epoch_val_global_acc > best_val_acc:
                best_val_acc = epoch_val_global_acc
                best_epoch = epoch + 1
                torch.save(self.state_dict(), model_path)

            message = (
                f"Epoch: {epoch + 1}"
                + f"\nTRAIN | loss: {epoch_train_loss:.4f}"
                + f" - global_acc: {epoch_train_global_acc:.4f}"
                + f" - class_acc: ["
                + " ".join([f"{a.item():.4f}" for a in epoch_train_class_acc])
                + "]"
                + f" \nVALID | loss: {epoch_val_loss:.4f}"
                + f" - global_acc: {epoch_val_global_acc:.4f}"
                + f" - class_acc: ["
                + " ".join([f"{a.item():.4f}" for a in epoch_val_class_acc])
                + "]"
                + (" - BEST!" if best_epoch == epoch + 1 else "")
            )

            print(message)
