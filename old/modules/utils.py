"""vials-detection/modules/utils.py

Summary:
    Defines the utility functions of the vials-detection project.


Copyright 2021 - Filippo Guerranti <filippo.guerranti@student.unisi.it>

DO NOT REDISTRIBUTE. 
The software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY 
KIND, either expressed or implied.
"""
import argparse
import os
import pickle
from typing import List, Tuple, Union

import cv2
import torch
import torchvision
import tqdm


class ModuleSetup:
    def __init__(
        self, name: str, folder_path: str, regions_file: Union[str, None] = None
    ):
        """Handles the setup of a given processing step.

        The ModuleSetup class is used to initialize the regions of the module.
        It is assumed that each module in the manifacturing process has fixed regions in which
        the vials are located. This assumption allows us to use a reference image to define
        the regions of interest and use them to select all the vials of the given module.

        Args:
            name (str):
                Module name.
            folder_path (str):
                Module folder.
            regions_file (Union(str, None), optional):
                Regions filepath. If defined and present, regions are loaded from the file.
                If defined and not present, regions are defined by the user and stored in the file.
                Default: None.
        """
        self.name = name
        self.folder = os.path.abspath(folder_path)
        if os.path.exists(regions_file) and os.path.isfile(regions_file):
            self.regions = pickle.load(open(regions_file, "rb"))
        else:
            self.regions = None
            self.set_regions_from_reference()
            pickle.dump(self.regions, open(regions_file, "wb"))

        self.crop_by_regions()

    def set_regions_from_reference(self, img_path: str = None):
        """Sets the ROIs given a reference image.

        A popup window is opened and the user can manually set the regions of interest by
        drawing them using the mouse.
        Click:
            - 'R' to (R)eset the window.
            - 'L' to remove the (L)ast drawn box.
            - 'ENTER' to confirm the choices.

        Args:
            img_path (str, optional):
                Path to the reference image. Default: None.
        """
        global ix, iy, regions, imgs
        ix, iy = -1, -1
        regions = []
        imgs = []

        def draw_rectangle(event, x, y, flag, par):
            global ix, iy, regions, imgs

            # if mouse left button is clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                ix, iy = x, y

            # if mouse left button is released
            elif event == cv2.EVENT_LBUTTONUP:
                img = imgs[-1].copy()
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
                regions.append([(ix, iy), (x, y)])
                imgs.append(img)
                cv2.imshow("Reference image", imgs[-1])

        if img_path is None:
            img_path = os.path.join(self.folder, os.listdir(self.folder)[0])

        original_img = cv2.imread(img_path)

        # different copies of the image are stored to provide "delete last" functionality
        imgs.append(original_img.copy())

        cv2.namedWindow(winname="Reference image")
        cv2.setMouseCallback("Reference image", draw_rectangle)

        while True:
            cv2.imshow("Reference image", imgs[-1])
            key = cv2.waitKey(1) & 0xFF

            # press 'r' to reset the window
            if key == ord("r"):
                imgs = [original_img]
                regions = []

            # press 'l' to remove last box
            elif key == ord("l"):
                if len(imgs) > 0 and len(regions) > 0:
                    imgs.pop()
                    regions.pop()
                else:
                    imgs = [original_img]
                    regions = []

            # press 'ENTER' to confirm
            elif key == 13:
                break

        cv2.destroyAllWindows()
        self.regions = regions

    @staticmethod
    def __crop(original_img, region):
        p1, p2 = region
        top_left_corner = (min(p1[0], p2[0]), min(p1[1], p2[1]))
        bottom_right_corner = (max(p1[0], p2[0]), max(p1[1], p2[1]))
        cropped_img = original_img[
            top_left_corner[1] : bottom_right_corner[1],
            top_left_corner[0] : bottom_right_corner[0],
        ].copy()
        return cropped_img

    def crop_by_regions(self):
        assert (
            self.regions is not None
        ), "Reference regions must be set, call set_regions_by_reference()."

        cropped_folder = os.path.join(
            os.path.abspath(os.path.join(self.folder, os.pardir)), "cropped"
        )

        if not os.path.exists(cropped_folder):
            os.mkdir(cropped_folder)

        for img_name in os.listdir(self.folder):
            if img_name.endswith(".bmp"):
                original_img = cv2.imread(os.path.join(self.folder, img_name))
                img_name_splits = img_name.split(".")
                assert (
                    len(img_name_splits) == 2
                ), f"Incorrect image name: {img_name}. Too many splits."
                img_name_no_ext, ext = img_name.split(".")[0], img_name.split(".")[-1]
                for i, region in enumerate(self.regions):
                    cropped_img = self.__crop(original_img, region)
                    cropped_img_name = os.path.join(
                        cropped_folder, img_name_no_ext + f"_{i}." + ext
                    )
                    cv2.imwrite(cropped_img_name, cropped_img)


class DataAugmentation:
    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]] = (-5.0, 5.0),
        crop_resize: Tuple[int, int] = (90, 90),
        crop_scale: Tuple[float, float] = (0.7, 0.9),
        brightness: Union[float, Tuple[float, float]] = (0.8, 1.5),
        contrast: Union[float, Tuple[float, float]] = (0.7, 1.8),
        saturation: Union[float, Tuple[float, float]] = (0.2, 1.8),
        hue: Union[float, Tuple[float, float]] = (-0.5, 0.5),
        horizontal_flip_prob: float = 0.5,
        gaussian_noise: Tuple[float, float] = (0.0, 0.01),
        gaussian_noise_prob: float = 0.5,
    ):
        """Data augmentation class.

        Applies several transformation to the input image:
        - RandomRotation
        - RandomResizeCrop
        - ColorJitter
        - RandomHorizontalFlip
        - RandomGaussianNoise

        Args:
            degrees (Union[float, Tuple[float, float]], optional):
                Range of degrees to select from. If degrees is a number instead of sequence
                like (min, max), the range of degrees will be (-degrees, +degrees).
                Default: (-5.0, 5.0).
            crop_resize (Tuple[int, int], optional):
                Expected output size of the crop, for each edge. If size is an int instead
                of sequence like (h, w), a square output size (size, size) is made.
                If provided a sequence of length 1, it will be interpreted as
                (size[0], size[0]). Default: (90, 90).
            crop_scale (Tuple[float, float], optional):
                Specifies the lower and upper bounds for the random area of the crop,
                before resizing. The scale is defined with respect to the area of the
                original image. Default: (0.7, 0.9).
            brightness (Union[float, Tuple[float, float]], optional):
                How much to jitter brightness. brightness_factor is chosen uniformly from
                [max(0, 1 - brightness), 1 + brightness] or the given [min, max].
                Should be non negative numbers. Default: (0.8, 1.5).
            contrast (Union[float, Tuple[float, float]], optional):
                How much to jitter contrast. contrast_factor is chosen uniformly from
                [max(0, 1 - contrast), 1 + contrast] or the given [min, max].
                Should be non negative numbers.Default: (0.7, 1.8).
            saturation (Union[float, Tuple[float, float]], optional):
                How much to jitter saturation. saturation_factor is chosen uniformly from
                [max(0, 1 - saturation), 1 + saturation] or the given [min, max].
                Should be non negative numbers. Default: (0.2, 1.8).
            hue (Union[float, Tuple[float, float]], optional):
                How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue]
                or the given [min, max].
                Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5. Default: (-0.5, 0.5).
            horizontal_flip_prob (float, optional):
                Probability of the image being flipped. Should be non negative number.
                Default: 0.5.
            gaussian_noise (Tuple[float, float], optional):
                Mean and standard deviation of the gaussian distribution for the noise. It
                is expected to be a tuple of two elements, interpreted as (mean, stddev).
                Default: (0.0, 0.01).
            gaussian_noise_prob (float, optional):
                Probability of the random gaussian noise being applied to the image.
                Should be non negative number. Default: 0.5.
        """

        self.augmentation = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomRotation(degrees=degrees),
                torchvision.transforms.RandomResizedCrop(
                    size=crop_resize, scale=crop_scale
                ),
                torchvision.transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                ),
                torchvision.transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
                torchvision.transforms.RandomApply(
                    [GaussianNoise(mean=gaussian_noise[0], stddev=gaussian_noise[1])],
                    p=gaussian_noise_prob,
                ),
            ]
        )

    def __call__(self, tensor):
        """Call the class on an input tensor.

        Args:
            tensor (torch.Tensor):
                Input tensor.

        Returns:
            output_tensor (torch.Tensor):
                Output tensor, with added gaussian noise.
        """
        return self.augmentation(tensor)


class GaussianNoise:
    def __init__(
        self,
        mean: float = 0.0,
        stddev: float = 0.01,
    ) -> None:

        """GaussianNoise transform.

        Add gaussian noise to an image, given the mean and standard deviation of the gaussian
        distribution.

        Args:
            mean (float, optional):
                Mean of the distribution. Default: 0.0.
            stddev (float, optional):
                Standard deviation of the distribution. Default: 0.01.
        """

        self.stddev = stddev
        self.mean = mean

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Call the class on an input tensor.

        Args:
            tensor (torch.Tensor):
                Input tensor.

        Returns:
            output_tensor (torch.Tensor):
                Output tensor, with added gaussian noise.
        """
        out = tensor + torch.randn(tensor.size()) * self.stddev + self.mean
        return (out - out.min()) / (out.max() - out.min())


class SAELoss(torch.nn.modules.loss._WeightedLoss):
    __constants__ = ["reduction", "supervision_weight", "reconstruction_weight"]

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        supervision_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
    ) -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        self.supervision_weight = supervision_weight
        self.reconstruction_weight = reconstruction_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = input[0]
        labels = target[0]

        reconstructed_images = input[1]
        original_images = target[1]

        return self.supervision_weight * torch.nn.functional.cross_entropy(
            logits, labels, reduction=self.reduction
        ) + self.reconstruction_weight * torch.nn.functional.mse_loss(
            reconstructed_images, original_images, reduction="mean"
        )


def main_args_parser() -> argparse.Namespace:
    """Command-line interface.

    Create three main subparser (train, eval, classify) and add the appropriated arguments
    for each subparser.

    Return:
        args (argparse.Namespace):
            Command line input arguments
    """

    parser = argparse.ArgumentParser(description="Pharmaceutical vials detection")

    parser.add_argument(
        "model",
        help="<required> model to be trained",
        metavar="MODEL_NAME",
        choices=["CNN", "SAE"],
    )

    parser.add_argument(
        "dataset",
        metavar="DATASET_DIR",
        type=str,
        help="<required> path to input dataset directory",
    )

    parser.add_argument(
        "-c",
        "--classes",
        nargs="+",
        type=str,
        help='(default=["absent", "blank", "cap", "stopper"]) list of dataset classes',
        default=["absent", "blank", "cap", "stopper"],
        metavar="CLASS",
    )

    parser.add_argument(
        "-s",
        "--splits",
        nargs=3,
        type=float,
        help="(default=[0.6,0.2,0.2]) proportions for the dataset split into training and validation set",
        default=[0.6, 0.2, 0.2],
        metavar=("TRAIN", "VAL", "TEST"),
    )

    parser.add_argument(
        "-a",
        "--augmentation",
        action="store_true",
        help="set data augmentation procedure ON",
    )

    parser.add_argument(
        "-A",
        "--test_augmentation",
        action="store_true",
        help="set test data augmentation procedure ON",
    )

    parser.add_argument(
        "-C",
        "--encoded_size",
        type=int,
        help="(default=70) size of the encoded space",
        default=70,
    )

    parser.add_argument(
        "--reconstruction_weight",
        type=float,
        help="(default=0.8) how much to weight the reconstruction loss",
        default=0.8,
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="(default=64) number of sample per batch",
        default=64,
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="(default=1) number of training epochs",
        default=1,
    )

    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        help="(default=0.0001) learning rate",
        default=0.0001,
    )

    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        help="(default=3) number of workers",
        default=3,
    )

    parser.add_argument(
        "-d",
        "--device",
        type=str,
        help="(default=cuda:0) device to be used for computations {cpu, cuda:0, cuda:1, ...}",
        default="cuda:0",
    )

    parser.add_argument(
        "-D",
        "--datetime",
        type=str,
        help="(default: None) run datetime, for model name",
        default="None",
    )

    args = parser.parse_args()

    return args


def closest_factors(n: int) -> Tuple[int, int]:
    """Compute the closest integers factors of a number.

    Given a number n, its closest integers factors are those numbers which are factors
    of n and that, if multiplied together give n itself. Additionaly, they are the closest
    possible.

    Example:
        >>> closest_factors(60)
        6, 10
        >>> closest_factors(20)
        4, 5

    Args:
        n (int):
            Input number.

    Returns:
        closest_factors (Tuple[int, int]):
            The closest factors of n.
    """

    from math import sqrt

    assert isinstance(n, int), f"num must be integer"
    a = sqrt(n)
    if n % a == 0:
        return int(a), int(a)
    else:
        a = int(a)
        if n % a == 0:
            return a, int(n / a)
        else:
            for i in range(1, n - a):
                if n % (a + i) == 0:
                    return a + i, int(n / (a + i))


def mean_stddev(dataset):
    data_loader = dataset.get_loader(batch_size=2048, num_workers=12)

    sum = torch.zeros((3,))
    sum_squares = torch.zeros((3,))
    count = torch.zeros((3,))
    for X, Y in tqdm.tqdm(data_loader):
        X_copy = torch.clone(X)
        count += torch.tensor([int(X_copy.numel() / 3)] * 3)
        shape = X_copy.shape
        X_vectorized = X_copy.reshape((shape[0], shape[1], -1))
        X_vectorized_squared = X_vectorized ** 2
        sum += X_vectorized.sum(dim=-1).sum(dim=0)
        sum_squares += X_vectorized_squared.sum(dim=-1).sum(dim=0)

    mean = sum / count
    stddev = torch.sqrt((sum_squares / count) - (sum / count) ** 2)

    return mean, stddev
