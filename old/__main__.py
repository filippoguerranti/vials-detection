import os
import sys
import tqdm
import pathlib
import datetime

import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt

from modules import data, models
from utils import utils

if __name__ == "__main__":

    args = utils.parse_arg()

    dataset = data.VialsDataset(
        dir=args.dataset, classes=["absent", "blank", "cup", "stopper"]
    )

    dataset = data.DataProcessor(
        dir=pathlib.Path(args.dataset),
        batch_size=args.batch_size,
        img_size=(28, 28),
    )
    dataset.load()

    class_count = {i: 0 for i, _class in enumerate(dataset.classes)}
    for i in tqdm.tqdm(range(1000)):
        batch = dataset.get_batch()
        batch_sum = tf.cast(tf.reduce_sum(batch[1], axis=0), dtype=tf.int32)
        for idx, val in enumerate(batch_sum):
            class_count[idx] += val.numpy()

    print(class_count)

    exit()
    if args.model.lower() == "autoencoder":
        model = models.Autoencoder()
    elif args.model.lower() == "cnn":
        model = models.SimpleCNN(data.classes.shape[0])
    else:
        raise NotImplementedError(f"{args.model} not implemented yet.")

    if args.checkpoint:
        model.load_weights(args.checkpoint)

    else:
        model.compile(
            loss=tf.keras.losses.MSE(),
            metrics=["accuracy"],
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
        )

        checkpoint_path = (
            rf"checkpoints/{datetime.datetime.now().strftime('%d%m%y-%H%M%S')}.ckpt"
        )
        callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
        )

        history = model.fit(
            dataset,
            epochs=args.epochs,
            # batch_size=args.batch_size,
            # shuffle=True,
            # validation_data=(x_val, x_val),
            callbacks=[callback],
        )

    decoded_imgs = model.predict(dataset)
