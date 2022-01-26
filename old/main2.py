import os
import sys
import tqdm
import pathlib
import datetime

import torch

from modules import data, models
from utils import utils


def main():
    args = utils.parse_arg()

    dataset = data.VialsDataset(dir=args.dataset, classes=args.classes)

    train, val, test = dataset.split()

    train.statistics()
    val.statistics()
    test.statistics()


if __name__ == "__main__":
    main()
