import collections
import os
from typing import List, Tuple, Union

import numpy as np
import PIL
import torch
import torchvision


class VialsDataset(torch.utils.data.Dataset):
    """Vials Dataset.

    Initializes both:
        - the overall dataset, starting from a directory having subdirectory with
        classes names, each one containing the images of the relative class;
        - the splitted dataset, starting from an array of images paths and labels.

    If classes is provided, then the dataset will only be made of images of the
    selected classes, otherwise classes are inferred by either the directory or the
    data array.

    Args:
        dir (Union[str, None], optional):
            Images directory path. If None, the dataset is supposed to be used as a
            split of the overall one. Default: None.

        data (Union[np.array, None], optional):
            Array of shape (num_samples, 2), where each row contains the image path
            and the relative class. It is generally used to instantiate a splitted
            dataset. Default: None.

        classes (Union[List[str], None], optional):
            List of classes to be considered for the task. Default: None.

        imgs_channels (Union[int, None]):
            Number of channels for the dataset images. If None, the default size is
            kept, otherwise images are converted accordingly.

    Note:
        dir and data are mutually exclusive.
    """

    def __init__(
        self,
        dir: Union[str, None] = None,
        data: Union[np.array, None] = None,
        classes: Union[List[str], None] = None,
        imgs_channels: int = 3,
        imgs_size: int = 32,
    ) -> None:

        if not ((dir is None) ^ (data is None)):
            raise ValueError(f"not mutually exclusive [dir = {dir}, data = {data}]")

        # initialization
        self.imgs_dir = dir
        if imgs_size == 32 or imgs_size == 90:
            self.imgs_size = (imgs_size, imgs_size)
        else:
            raise ValueError(
                f"images size must be either 32 or 90: imgs_size = {imgs_size}"
            )

        self.augmentation = None
        if imgs_channels == 1 or imgs_channels == 3:
            self.imgs_channels = imgs_channels
        else:
            raise ValueError(
                f"number of channels must be either 1 or 3: imgs_channels = {imgs_channels}"
            )
        # create splitted dataset
        if self.imgs_dir is None and data is not None:

            if classes is None:  # classes inferred from data
                classes = [_class for _class in np.unique(data[:, 1])]

            else:  # classes provided as input
                # TODO:
                # - what if classes are provided but some of them are not in the
                #   data array? (unrecognized classes)
                # - what if some of the classes in data array are not in the
                #   provided ones? (very rare situation)
                for _class in classes:
                    if _class not in [_class_ for _class_ in np.unique(data[:, 1])]:
                        raise ValueError(f"unrecognized class '{_class}'")

        # create overall dataset
        elif self.imgs_dir is not None and data is None:
            if os.path.isfile(dir):
                raise NotADirectoryError(f"invalid data path {dir}")
            if not os.path.exists(dir):
                os.makedirs(dir)

            if classes is None:  # classes inferred from the directory
                classes = [
                    _class
                    for _class in os.listdir(self.imgs_dir)
                    if not _class.startswith(".")
                ]
            else:  # classes provided as input
                for _class in classes:
                    if _class not in [
                        _class_
                        for _class_ in os.listdir(self.imgs_dir)
                        if not _class_.startswith(".")
                    ]:
                        raise ValueError(f"unrecognized class '{_class}'")

            data = np.array(
                sorted(
                    [
                        [os.path.join(self.imgs_dir, _class, img_path), _class]
                        for _class in os.listdir(self.imgs_dir)
                        if not _class.startswith(".") and _class in classes
                        for img_path in os.listdir(os.path.join(self.imgs_dir, _class))
                        if img_path.endswith(".png")
                    ]
                ),
                dtype=np.str,
            )

        else:
            raise ValueError(f"not mutually exclusive [dir = {dir}, data = {data}")

        self.classes = classes
        self._setup(data)
        self.mean = (0.5041, 0.4538, 0.4334)
        self.stddev = (0.3195, 0.3312, 0.3299)

    def __len__(self) -> int:
        """Dataset length"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the dataset item (image-label pair) given the index.

        Args:
            idx (int):
                Sample index.

        Return:
            image (torch.Tensor):
                Image at index idx, that is a tensor of shape [n_channels, height, width]
                where n_channels, height and width are provided in the initialization of
                the dataset instance.

            label (torch.Tensor):
                Image label. It is a tensor of shape [1,] representing the image class.

        """
        preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.stddev, inplace=True),
            ]
        )
        image = PIL.Image.open(self.data[idx][0]).convert("RGB")
        image = preprocess(image)

        if self.augmentation is not None:
            image = self.augmentation(image)

        postprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.imgs_size),
            ]
            + (
                [
                    torchvision.transforms.Grayscale(
                        num_output_channels=self.imgs_channels
                    )
                ]
                if self.imgs_channels == 1
                else []
            )
        )
        image = postprocess(image)

        label = self.class_name_to_label[self.data[idx][1]]

        return (image, torch.tensor(label).long())

    def _setup(self, data: np.array) -> None:
        """Set the dataset attributes given the array of images paths and labels.

        Args:
            data (np.array):
                Array of shape [num_samples, 2], where each row contains the image path
                and the relative class.
        """

        self.data = data

        self.classes_counts = collections.Counter(self.data[:, 1])

        self.class_name_to_label = {
            _class: label for label, _class in enumerate(self.classes_counts.keys())
        }

        self.label_to_class_name = {
            label: _class for label, _class in enumerate(self.classes_counts.keys())
        }

        self.classes_distributions = {
            _class: count / sum(self.classes_counts.values())
            for _class, count in self.classes_counts.items()
        }

        self.class_to_imgs_ids = {
            _class: np.squeeze(np.array(np.where(self.data[:, 1] == _class)))
            for _class in self.classes
        }

        weights = [1.0 / count for _, count in self.classes_counts.items()]
        self.sampler_weights = torch.from_numpy(
            np.array(
                [
                    weights[self.class_name_to_label[_class]]
                    for _class in self.data[:, 1]
                ]
            )
        ).double()

    def set_augmentation(
        self,
        operations: Union[torch.nn.Sequential, torchvision.transforms.Compose],
    ) -> None:
        """Set custom data augmentation operations to be applied to each sample.

        Args:
            operations (Union[torch.nn.Sequential, torchvision.transforms.Compose]):
                Data augmentation operation to be applied to each sample.

        Example:
            >>> torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomRotation(30),
                    torchvision.transforms.RandomResizedCrop(28, scale=(0.9, 1.1)),
                ]
            )
        """
        self.augmentation = operations

    def split(
        self,
        proportions: list = [0.6, 0.2, 0.2],
        shuffle: bool = True,
        stratified: bool = True,
    ) -> List[torch.utils.data.Dataset]:
        """Split dataset into train, validation and test sets according to proportions.

        Args:
            proportions (list, optional):
                List of dataset proportions. The first number corresponds to the proportion
                of samples to be used as training set, the second number and the third one
                are to be used as proportions for the validation and test sets.
                Default: [0.6, 0.2, 0.2].

            shuffle (bool, optional):
                If True the dataset is shuffled and then splitted. Default: True.

            stratified (bool, optional):
                If True the distribution of the classes is maintained across the splits.
                Default: True.

        Returns:
            datasets list(data.VialsDataset):
                List of splitted datasets.
        """
        if len(self.data) == 0:
            raise RuntimeError("empty dataset cannot be splitted")

        if not sum(proportions) == 1.0:
            raise ValueError("proportions must sum up to 1")
        if not all([p > 0.0 for p in proportions]):
            raise ValueError("proportions must be greater than zero")
        if len(proportions) < 2 or len(proportions) > 3:
            raise ValueError("proportions must be either 2 or 3 values")

        num_splits = len(proportions)

        if stratified:
            indices = [[] for _ in range(num_splits)]
            for (_, images) in self.class_to_imgs_ids.items():
                perm = (
                    torch.randperm(len(images))
                    if shuffle
                    else torch.arange(len(images))
                )
                perm_images = images[perm]
                start = 0
                for i, p in enumerate(proportions):
                    num_samples = (
                        int(p * len(images)) if int(p * len(images)) != 0 else 1
                    )
                    end = start + num_samples if i < num_splits - 1 else len(images)
                    indices[i].append(perm_images[start:end].astype(np.int64))
                    start = end
            indices = list(map(lambda x: np.concatenate(x, axis=0), indices))

        else:
            indices = []
            perm = torch.randperm(len(self)) if shuffle else torch.arange(len(self))
            start = 0
            for i, p in enumerate(proportions):
                num_samples = int(p * len(self)) if int(p * len(self)) != 0 else 1
                end = start + num_samples if i < num_splits - 1 else len(self)
                indices.append(perm[start:end])
                start = end

        datasets = [
            VialsDataset(
                dir=None,
                data=np.take(self.data, indices=i, axis=0),
                classes=self.classes,
                imgs_size=self.imgs_size[0],
                imgs_channels=self.imgs_channels,
            )
            for i in indices
        ]

        return datasets

    def get_loader(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        weighted_sampler: bool = False,
        shuffle: bool = False,
    ) -> torch.utils.data.DataLoader:
        """Returns the DataLoader for the current dataset.

        Args:
            batch_size (int, optional):
                Number of samples per batch to be loaded. Default: 64.
            num_workers (int, optional):
                Number of subprocesses to use for data loading. If num_workers == 0, data
                will be loaded in the main process. Default: 0.
            weighted_sampler (bool, optional):
                If True, the WeightedRandomSampler is used to load batches. Default: True.
            shuffle (bool, optional):
                If True, data is randomly shuffled at every epoch. Default: False.

        Returns:
            data_loader (torch.utils.data.DataLoader):
                Iterable over the current dataset.
        """

        if len(self.data) == 0:
            raise RuntimeError("empty dataset cannot be loaded")

        if shuffle and weighted_sampler:
            raise RuntimeError("shuffle and sampler cannot work simultaneously")

        if shuffle:
            data_loader = torch.utils.data.DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            )
        elif weighted_sampler:
            sampler = torch.utils.data.WeightedRandomSampler(
                self.sampler_weights, len(self.sampler_weights)
            )
            data_loader = torch.utils.data.DataLoader(
                self,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=sampler,
            )
        else:
            data_loader = torch.utils.data.DataLoader(
                self,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        return data_loader

    def statistics(self) -> None:
        """Prints some basic statistics of the current dataset."""

        print("N. samples:    \t{0}".format(len(self.data)))
        print("Classes:       \t{0}".format(set(self.classes)))
        print(
            "Classes distr.: {0}".format(
                [
                    (_class, round(value, 4))
                    for _class, value in self.classes_distributions.items()
                ]
            )
        )

    def show_samples(
        self, num_samples: int = 5, save_path: Union[str, None] = None
    ) -> None:
        """Show dataset samples.

        Args:
            num_samples (int, optional):
                Number of samples to be shown. Default: 5.

            save_path (Union[str, None], optional):
                Save the images to the provided path. Default: None.
        """
        import matplotlib.pyplot as plt

        classes = list(self.class_to_imgs_ids.keys())

        images_ids = []
        for i in range(num_samples):
            class_ = classes[i % len(classes)]
            ids_ = self.class_to_imgs_ids[class_]
            id_ = ids_[torch.randperm(len(ids_))[0].item()]
            images_ids.append(id_)
        samples = [self.__getitem__(id_) for id_ in images_ids]

        width = 10
        plt.figure(figsize=(width, width / 5), constrained_layout=True)
        for i in range(num_samples):
            image = samples[i][0].numpy().swapaxes(0, 1).swapaxes(1, 2)
            label = self.label_to_class_name[torch.argmax(samples[i][1]).item()]
            ax = plt.subplot(1, num_samples, i + 1)
            plt.imshow(image, cmap="gist_gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title(label)
        fig = plt.gcf()
        plt.show()

        # save plot
        if save_path is not None and isinstance(save_path, str):
            BASEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            FOLDER = os.path.join(BASEDIR, "results/images")
            if not os.path.exists(FOLDER):
                os.makedirs(FOLDER)
            FILEPATH = os.path.join(FOLDER, save_path)
            fig.savefig(f"{FILEPATH}.png", dpi=1000)


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
                    [
                        self.GaussianNoise(
                            mean=gaussian_noise[0], stddev=gaussian_noise[1]
                        )
                    ],
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
