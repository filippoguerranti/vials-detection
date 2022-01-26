"""vials-detection/modules/data.py

Summary:
    Defines the dataset class and related functions.

Classes:
    VialsDataset(torch.utils.data.Dataset)

Methods:
    VialsDataset.__init__()
    VialsDataset.__len__()
    VialsDataset.__getitem__()
    VialsDataset._setup()
    VialsDataset.set_augmentation()
    VialsDataset.split()
    VialsDataset.get_loader()
    VialsDataset.statistics()
    VialsDataset.show_samples()


Copyright 2021 - Filippo Guerranti <filippo.guerranti@student.unisi.it>

DO NOT REDISTRIBUTE. 
The software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY 
KIND, either expressed or implied.
"""
import collections
import os
from typing import List, Tuple, Union

import numpy as np
import PIL
import torch
import torchvision


class VialsDataset(torch.utils.data.Dataset):
    """
    Attributes:
        `imgs_dir` (`str`):
            Images directory path.
        `classes` (`list`):
            List of classes to be considered for the task.
        `classes_counts` (`collections.Counter`):
            Classes names as keys and the respective number of elements as values.
        `class_name_to_label` (`dict`):
            Classes names as keys and the respective label as values.
        `label_to_class_name` (`dict`):
            Classes label as keys and the respective name as values.
        `classes_distributions` (`dict`):
            Classes names as keys and the respective proportion as values.
        `class_to_imgs_ids` (`dict`):
            Classes names as keys and the respective images indices as values.
        `augmentation` (`bool`):
            If not None, data augmentation operations are applied to each sample.

    Methods:
        `set_preprocess()`:
            Set a custom preprocess operation to be applied to each sample.
        `split()`:
            Split dataset into train, validation and test sets according to proportions.
        `get_loader()`:
            Returns the `DataLoader` for the current dataset.
        `statistics()`:
            Prints some basic statistics of the current dataset.
    """

    def __init__(
        self,
        dir: Union[str, None] = None,
        data: Union[np.array, None] = None,
        classes: Union[List[str], None] = None,
        imgs_channels: Union[int, None] = None,
    ) -> None:
        """`VialsDataset` constructor.

        Initializes both:
            - the overall dataset, starting from a directory having subdirectory with
            classes names, each one containing the images of the relative class;
            - the splitted dataset, starting from an array of images paths and labels.

        If classes is provided, then the dataset will only be made of images of the
        selected classes, otherwise classes are inferred by either the directory or the
        data array.

        Args:
            `dir` (`Union[str, None]`, optional):
                Images directory path. If None, the dataset is supposed to be used as a
                split of the overall one. Default: `None`.

            `data` (`Union[np.array, None]`, optional):
                Array of shape (num_samples, 2), where each row contains the image path
                and the relative class. It is generally used to instantiate a splitted
                dataset. Default: `None`.

            `classes` (`Union[List[str]`, None], optional):
                List of classes to be considered for the task. Default: `None`.

            `imgs_channels` (`Union[int, None]`):
                Number of channels for the dataset images. If None, the default size is
                kept, otherwise images are converted accordingly.

        Note:
            `dir` and `data` are mutually exclusive.
        """

        if not ((dir is None) ^ (data is None)):
            raise ValueError(f"not mutually exclusive [dir = {dir}, data = {data}]")

        # initialization
        self.imgs_dir = dir
        self.imgs_size = (32, 32)

        self.augmentation = None
        self.imgs_channels = imgs_channels

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
            `idx` (`int`):
                Sample index.

        Return:
            `image` (`torch.Tensor`):
                Image at index idx, that is a tensor of shape `[height, width]` where
                `height` and `width` are provided in the initialization of the dataset
                instance.

            `one_hot_label` (`torch.Tensor`):
                One hot representation of the image label. It is a tensor of shape
                `[num_classes]` where `num_classes` is the number of dataset classes.

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
        )
        image = postprocess(image)

        label = self.class_name_to_label[self.data[idx][1]]

        return (
            image,
            torch.tensor(label).long()
            # torch.nn.functional.one_hot(
            #     torch.tensor(label), num_classes=len(self.classes)
            # ).float(),
        )

    def _setup(self, data: np.array) -> None:
        """Set the dataset attributes given the array of images paths and labels.

        Args:
            `data` (`np.array`):
                Array of shape `[num_samples, 2]`, where each row contains the image path
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
            `operations` (`Union[torch.nn.Sequential, torchvision.transforms.Compose]`):
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
        keep_classes_distributions: bool = True,
    ) -> List[torch.utils.data.Dataset]:
        """Split dataset into train, validation and test sets according to proportions.

        Args:
            `proportions` (`list`, optional):
                List of dataset proportions. The first number corresponds to the proportion
                of samples to be used as training set, the second number and the third one
                are to be used as proportions for the validation and test sets.
                Default: `[0.6, 0.2, 0.2]`.

            `shuffle` (`bool`, optional):
                If `True` the dataset is shuffled and then splitted. Default: `True`.

            `keep_classes_distributions` (`bool`, optional):
                If `True` the distribution of the classes is maintained across the splits.
                Default: `True`.

        Returns:
            `datasets` `list(data.VialsDataset)`:
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

        if keep_classes_distributions:
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
        """Returns the `DataLoader` for the current dataset.

        Args:
            `batch_size` (`int`, optional):
                Number of samples per batch to be loaded. Default: `64`.
            `num_workers` (`int`, optional):
                Number of subprocesses to use for data loading. If `num_workers == 0`, data
                will be loaded in the main process. Default: `0`.
            `weighted_sampler` (bool, optional):
                If `True`, the `WeightedRandomSampler` is used to load batches. Default: `True`.
            `shuffle` (`bool`, optional):
                If `True`, data is randomly shuffled at every epoch. Default: `False`.

        Returns:
            `data_loader` (`torch.utils.data.DataLoader`):
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
            `num_samples` (`int`, optional):
                Number of samples to be shown. Default: `5`.

            `save_path` (`Union[str, None]`, optional):
                Save the images to the provided path. Default: `None`.
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
