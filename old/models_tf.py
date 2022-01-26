import tensorflow as tf


class SimpleCNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.models_layers = []

        # Block 1
        self.models_layers.append(tf.keras.layers.Conv2D(32, 5, activation="relu"))
        self.models_layers.append(tf.keras.layers.MaxPooling2D())
        self.models_layers.append(tf.keras.layers.BatchNormalization())

        # Block 2
        self.models_layers.append(tf.keras.layers.Conv2D(64, 5, activation="relu"))
        self.models_layers.append(tf.keras.layers.MaxPooling2D())
        self.models_layers.append(tf.keras.layers.BatchNormalization())

        # Block 3
        self.models_layers.append(tf.keras.layers.Conv2D(128, 3, activation="relu"))
        self.models_layers.append(tf.keras.layers.MaxPooling2D())
        self.models_layers.append(tf.keras.layers.BatchNormalization())

        # Block 4
        # self.models_layers.append(tf.keras.layers.Conv2D(256, 3, activation="relu"))
        # self.models_layers.append(tf.keras.layers.MaxPooling2D())
        # self.models_layers.append(tf.keras.layers.BatchNormalization())

        # Output block
        self.models_layers.append(tf.keras.layers.Dropout(0.3))
        self.models_layers.append(tf.keras.layers.Flatten())
        self.models_layers.append(
            tf.keras.layers.Dense(num_classes, activation="softmax")
        )

    def call(self, input_tensor, training=False):
        for idx, layer in enumerate(self.models_layers):
            if idx == 0:
                x = layer(input_tensor)
            else:
                x = layer(x)
        return x


class Autoencoder(tf.keras.Model):
    def __init__(self, dropout: float = None):
        super(Autoencoder, self).__init__()

        self.encoder_layers = []
        self.decoder_layers = []

        # Encoder
        self.encoder_layers.append(
            tf.keras.layers.Conv2D(16, 5, padding="same", activation="relu")
        )
        self.encoder_layers.append(tf.keras.layers.MaxPooling2D(2, padding="same"))
        self.encoder_layers.append(tf.keras.layers.Conv2D(32, 3, activation="relu"))
        self.encoder_layers.append(tf.keras.layers.MaxPooling2D(2))
        self.encoder_layers.append(tf.keras.layers.Conv2D(64, 3, activation="relu"))
        self.encoder_layers.append(tf.keras.layers.MaxPooling2D(2))

        # Latent representation
        self.encoder_layers.append(tf.keras.layers.Flatten())

        # Decoder
        if dropout:
            self.decoder_layers.append(tf.keras.layers.Dropout(rate=0.3))

        self.decoder_layers.append(tf.keras.layers.Reshape((2, 2, -1)))
        self.decoder_layers.append(tf.keras.layers.UpSampling2D(2))
        self.decoder_layers.append(
            tf.keras.layers.Conv2DTranspose(32, 3, activation="relu")
        )
        self.decoder_layers.append(tf.keras.layers.UpSampling2D(2))
        self.decoder_layers.append(
            tf.keras.layers.Conv2DTranspose(16, 3, activation="relu")
        )
        self.decoder_layers.append(tf.keras.layers.UpSampling2D(2))
        self.decoder_layers.append(
            tf.keras.layers.Conv2DTranspose(1, 3, padding="same", activation="relu")
        )

    def call(self, input_tensor, training=False):
        for idx, layer in enumerate(self.encoder_layers):
            if idx == 0:
                encoded_representation = layer(input_tensor)
            else:
                encoded_representation = layer(encoded_representation)
        output = encoded_representation
        for idx, layer in enumerate(self.decoder_layers):
            output = layer(output)
        return output
