import argparse
import models


def parse_args() -> argparse.Namespace:

    model_names = sorted(
        name
        for name in models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(models.__dict__[name])
    )

    parser = argparse.ArgumentParser(description="Pharmaceutical vials classification")
    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="cnn32",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: cnn32)",
    )
    parser.add_argument(
        "--encoded_size",
        type=int,
        help="size of the encoded space (default: 20)",
        default=20,
        metavar="N",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=str,
        help='list of dataset classes (default: ["absent", "blank", "cap", "stopper"]) ',
        default=["absent", "blank", "cap", "stopper"],
        metavar="CLASS",
    )
    parser.add_argument(
        "--rw",
        "--reconstruction_weight",
        type=float,
        help="how much to weight the reconstruction loss (default: 0.8) ",
        default=1.0,
        metavar="N",
        dest="reconstruction_weight",
    )
    parser.add_argument(
        "--splits",
        nargs=3,
        type=float,
        help="proportions for the dataset split into training and validation set (default: [0.6,0.2,0.2]) ",
        default=[0.6, 0.2, 0.2],
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument(
        "--train_augmentation",
        action="store_true",
        help="set train data augmentation procedure ON",
    )
    parser.add_argument(
        "--valid_augmentation",
        action="store_true",
        help="set valid data augmentation procedure ON",
    )
    parser.add_argument(
        "--test_augmentation",
        action="store_true",
        help="set test data augmentation procedure ON",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=12,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 12)",
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        metavar="N",
        help="number of total epochs to run (default: 10)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size (default: 128)",
    )
    parser.add_argument(
        "--optimizer",
        default="adam",
        type=str,
        help="optimizer: adam | sgd (default: adam)",
        choices={"adam", "sgd"},
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        default=1e-4,
        type=float,
        metavar="LR",
        help="optimizer learning rate (default: 1e-4)",
        dest="learning_rate",
    )
    parser.add_argument(
        "--wd",
        "--weight_decay",
        default=1e-6,
        type=float,
        metavar="W",
        help="optimizer weight decay (default: 1e-6)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="M",
        help="optimizer momentum (default: 0.9)",
    )
    parser.add_argument(
        "-p",
        "--print_freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        metavar="MODEL",
        help="evaluate model on test set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained model",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="seed for initializing training (default: none)",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        type=int,
        help="GPU id to use: 0 | 1 | None (default: None)",
        choices=[0, 1, None],
    )
    parser.add_argument(
        "--imgs_size",
        default=32,
        type=int,
        help="images size: 32 | 90 (default: 32)",
        choices=[32, 90],
    )
    parser.add_argument(
        "--imgs_channels",
        default=3,
        type=int,
        help="images channels: 1 | 3 (default: 3)",
        choices=[1, 3],
    )

    return parser.parse_args()
