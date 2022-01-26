import argparse
import os
import random
import shutil
import time
import warnings
from typing import List, Union

import numpy as np
import torch
import torchmetrics
from torchsummary import summary

import models
import utils
from utils.definitions import CHECKPOINTS_DIR


def main():
    args = utils.parse_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cuda.deterministic = True
        torch.use_deterministic_algorithms(True)
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    best_mcc = -1.0

    # device setup
    args.device = (
        torch.device(f"cuda:{args.gpu}")
        if torch.cuda.is_available() and args.gpu is not None
        else torch.device("cpu")
    )

    print(f"=> using device: {args.device}")

    # create model
    if args.pretrained or args.evaluate:
        print(f"=> using pre-trained model '{args.arch}'")
    else:
        print(f"=> creating model '{args.arch}'")

    model = models.__dict__[args.arch](
        pretrained=(args.pretrained or args.evaluate),
        encoded_size=args.encoded_size,
        n_classes=len(args.classes),
        n_channels=args.imgs_channels,
    )

    print(model.state_dict)

    MODEL_NAME = fr"{args.arch}-{args.encoded_size}"
    print(
        f"=> model input shape: [{args.batch_size}, {args.imgs_channels}, {args.imgs_size}, {args.imgs_size}]"
    )
    summary(
        model,
        (args.imgs_channels, args.imgs_size, args.imgs_size),
        batch_size=args.batch_size,
        device=f"cpu",
    )
    model.to(args.device)

    # set criterion
    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    # set optimizer
    if args.optimizer.startswith("adam"):
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer.startswith("sgd"):
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=True,
        )
    else:
        raise ValueError(f"unrecognized optimizer: {args.optimizer}")

    print(f"=> loading dataset from '{args.data}'")
    dataset = utils.data_loading.VialsDataset(
        dir=args.data,
        classes=args.classes,
        imgs_size=args.imgs_size,
        imgs_channels=args.imgs_channels,
    )

    ds_train, ds_valid, ds_test = dataset.split(proportions=args.splits, shuffle=True)

    print(
        f"\t=> train data: [{len(ds_train)}, "
        + f"{ds_train[0][0].shape[0]}, "
        + f"{ds_train[0][0].shape[1]}, "
        + f"{ds_train[0][0].shape[2]}]"
        + f"\n\t=> valid data: [{len(ds_valid)}, "
        + f"{ds_valid[0][0].shape[0]}, "
        + f"{ds_valid[0][0].shape[1]}, "
        + f"{ds_valid[0][0].shape[2]}]"
        + f"\n\t=> test data: [{len(ds_test)}, "
        + f"{ds_test[0][0].shape[0]}, "
        + f"{ds_test[0][0].shape[1]}, "
        + f"{ds_test[0][0].shape[2]}]"
    )

    # Data augmentation
    data_augmentation = utils.data_loading.DataAugmentation()
    if args.train_augmentation:
        ds_train.set_augmentation(data_augmentation)
    if args.valid_augmentation:
        ds_valid.set_augmentation(data_augmentation)
    if args.test_augmentation:
        ds_test.set_augmentation(data_augmentation)

    # getting training data loader
    train_loader = ds_train.get_loader(
        batch_size=args.batch_size,
        num_workers=args.workers,
        weighted_sampler=True,
        shuffle=False,
    )

    # getting validation data loader
    valid_loader = ds_valid.get_loader(
        batch_size=args.batch_size,
        num_workers=args.workers,
        weighted_sampler=True,
        shuffle=False,
    )

    # getting validation data loader
    test_loader = ds_test.get_loader(
        batch_size=args.batch_size,
        num_workers=args.workers,
        weighted_sampler=False,
        shuffle=False,
    )

    train_metrics = torchmetrics.MetricCollection(
        {
            "mcc": torchmetrics.MatthewsCorrcoef(num_classes=len(args.classes)),
            "f1": torchmetrics.F1(num_classes=len(args.classes), average="macro"),
            "prec": torchmetrics.Precision(
                num_classes=len(args.classes), average="macro"
            ),
            "recall": torchmetrics.Recall(
                num_classes=len(args.classes), average="macro"
            ),
        }
    )

    valid_metrics = torchmetrics.MetricCollection(
        {
            "mcc": torchmetrics.MatthewsCorrcoef(
                num_classes=len(args.classes), compute_on_step=False
            ),
            "f1": torchmetrics.F1(
                num_classes=len(args.classes), average="macro", compute_on_step=False
            ),
            "prec": torchmetrics.Precision(
                num_classes=len(args.classes), average="macro", compute_on_step=False
            ),
            "recall": torchmetrics.Recall(
                num_classes=len(args.classes), average="macro", compute_on_step=False
            ),
        }
    )

    if args.evaluate:
        validate(test_loader, model, criterion, valid_metrics, args, phase="TEST")
        return

    for epoch in range(args.epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, train_metrics, epoch, args)

        # evaluate on validation set
        mcc = validate(valid_loader, model, criterion, valid_metrics, args, epoch=epoch)

        # remember best mcc and save checkpoint
        is_best = mcc > best_mcc
        best_mcc = max(mcc, best_mcc)

        if is_best:
            print("BEST!")

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_mcc": best_mcc,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            filedir=CHECKPOINTS_DIR,
            filename=MODEL_NAME,
        )

    validate(test_loader, model, criterion, valid_metrics, args, phase="TEST")


def train(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    metrics: Union[torchmetrics.Metric, torchmetrics.MetricCollection],
    epoch: int,
    args: argparse.Namespace,
):
    metrics = metrics.to(args.device)
    total_time = Meter("time", "6.3f")
    losses = Meter("loss", ".4e")
    progress = ProgressMeter(
        len(loader), batch_meters=[losses], end_meters=[total_time], phase="TRAIN"
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(loader):

        images = images.to(args.device)
        target = target.to(args.device)

        # compute output
        output = model(images)

        # handle SAE model outputs
        if args.arch.startswith("sae"):
            if isinstance(output, tuple):
                reconstruction_criterion = torch.nn.MSELoss().to(args.device)
                # measure loss
                loss = criterion(
                    output[0], target
                ) + args.reconstruction_weight * reconstruction_criterion(
                    output[1], images
                )
                output = output[0]
            else:
                raise RuntimeError(
                    f"Model sae forward method must return tuple (output, reconstructed)"
                )

        else:
            # measure loss
            loss = criterion(output, target)

        # measure accuracy and record loss
        batch_metrics = metrics(output, target)
        losses.update(loss.item(), images.shape[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        total_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(
                batch=i,
                metrics=printable_metrics(batch_metrics, "6.4f"),
                epoch=epoch,
                end=False,
            )

    progress.display(
        batch=None,
        metrics=printable_metrics(batch_metrics, "6.4f"),
        epoch=epoch,
        end=True,
    )
    metrics.reset()


def validate(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    metrics: Union[torchmetrics.Metric, torchmetrics.MetricCollection],
    args: argparse.Namespace,
    epoch: Union[int, None] = None,
    phase: str = "VALID",
):
    metrics = metrics.to(args.device)
    total_time = Meter("time", "6.3f")
    losses = Meter("loss", ".4e")
    progress = ProgressMeter(
        len(loader), batch_meters=[losses], end_meters=[total_time], phase=phase
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(loader):

            images = images.to(args.device)
            target = target.to(args.device)

            # compute output
            output = model(images)

            # handle SAE model outputs
            if args.arch.startswith("sae"):
                if isinstance(output, tuple):
                    reconstruction_criterion = torch.nn.MSELoss().to(args.device)
                    # measure loss
                    loss = criterion(
                        output[0], target
                    ) + args.reconstruction_weight * reconstruction_criterion(
                        output[1], images
                    )
                    output = output[0]
                else:
                    raise RuntimeError(
                        f"Model sae forward method must return tuple (output, reconstructed)"
                    )

            else:
                # measure loss
                loss = criterion(output, target)

            # measure accuracy and record loss
            metrics(output, target)
            losses.update(loss.item(), images.shape[0])

            # measure elapsed time
            total_time.update(time.time() - end)
            end = time.time()

        overall_metrics = metrics.compute()
        metrics.reset()
        progress.display(
            batch=None,
            metrics=printable_metrics(overall_metrics, "6.4f"),
            epoch=epoch,
            end=True,
        )
        mcc_avg = overall_metrics["mcc"]

    return mcc_avg


def save_checkpoint(
    state: dict, is_best: bool, filedir: str = "models/", filename: str = "checkpoint"
):
    extension = r".pth.tar"
    model = os.path.join(filedir, fr"{filename}{extension}")
    best_model = os.path.join(filedir, fr"best_{filename}{extension}")
    torch.save(state, model)
    if is_best:
        shutil.copyfile(model, best_model)


class Meter:
    def __init__(self, name: str, fmt: str = "f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def printable_average(self):
        fmtstr = f"{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})"
        return fmtstr

    def printable_total(self):
        fmtstr = f"{self.name} {self.sum:{self.fmt}} ({self.avg:{self.fmt}})"
        return fmtstr


class ProgressMeter:
    def __init__(
        self,
        num_batches: int,
        batch_meters: List[Meter],
        end_meters: List[Meter],
        phase: str,
    ):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.batch_meters = batch_meters
        self.end_meters = end_meters
        self.phase = phase

    def display(
        self,
        batch: Union[int, None] = None,
        metrics: Union[str, None] = None,
        epoch: Union[int, None] = None,
        end: bool = False,
    ):
        entries = [
            f"[{self.phase}]"
            + (f" Epoch: [{epoch+1}]" if epoch is not None else "")
            + (
                f" " + self.batch_fmtstr.format(batch)
                if batch is not None and not end
                else "[end epoch]"
            )
        ]
        entries += [meter.printable_average() for meter in self.batch_meters]
        if metrics is not None:
            entries += [metrics]
        if end:
            entries += [meter.printable_total() for meter in self.end_meters]
        print(" | ".join(entries))

    def _get_batch_fmtstr(self, num_batches: int):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def printable_metrics(metrics: dict, fmt: str = "f"):
    entries = [f"{name} {val:{fmt}}" for name, val in metrics.items()]
    return " | ".join(entries)


if __name__ == "__main__":
    main()
