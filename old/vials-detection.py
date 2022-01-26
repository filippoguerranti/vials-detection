"""vials-detection/vials-detection.py script

Summary:
    Main script for the execution of:
        - train
        - evaluate
        - classify

Functions:
    main()


Copyright 2021 - Filippo Guerranti <filippo.guerranti@student.unisi.it>

DO NOT REDISTRIBUTE. 
The software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY 
KIND, either expressed or implied.
"""

import datetime
from modules import execution
from utils import utils


def main():
    """Main function which calls one of the execution modes according to the parsed arguments.

    Raises:
        `ValueError`:
            If `args.mode` is not `train` or `eval` or `classify`.
    """
    args = utils.main_args_parser()
    print(args)

    execution.train(
        model=args.model,
        dataset_dir=args.dataset,
        classes=args.classes,
        splits=args.splits,
        augmentation=args.augmentation,
        test_augmentation=args.test_augmentation,
        encoded_size=args.encoded_size,
        reconstruction_weight=args.reconstruction_weight,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        device=args.device,
        datetime_=args.datetime,
    )


if __name__ == "__main__":
    main()
