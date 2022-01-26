import logging
import os
import glob
import shutil


def log_and_print(message):
    logging.info(message)
    print(message)


def create_classes_folders_from_images_and_labels_folder(
    images_folder: str, labels_folder: str
):
    images_paths = []
    for subfolder in sorted(os.listdir(images_folder)):
        path = os.path.join(images_folder, subfolder)
        if not subfolder.startswith(".") and os.path.isdir(path):
            for image in sorted(glob.glob(os.path.join(path, "*.png"))):
                images_paths.append(image)

    labels_str = read_labels_from_files(labels_folder)
    classes = set(labels_str)

    for class_name in classes:
        os.mkdir(os.path.join(images_folder, class_name))
    for image_label in zip(images_paths, labels_str):
        src = image_label[0]
        dst = os.path.join(images_folder, image_label[1])
        shutil.copy(src, dst)


@staticmethod
def read_labels_from_files(folder: str):
    """Read the labels from a folder of files containing them.

    Args:
        folder (str):
            Path to labels folder.

    Returns:
        labels (list):
            List of labels.
    """
    labels = []
    for file in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        log_and_print(f"Loading labels from {file} to tensor")
        with open(file, "r") as f:
            for line in f.readlines():
                label = line.rstrip()
                labels.append(label)
    return labels
