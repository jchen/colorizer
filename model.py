import hyperparameters as hp

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Dropout,
    Flatten,
    Dense,
    UpSampling2D,
    Reshape,
)


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

        self.architecture = [
            Conv2D(64, (3, 3), activation="relu", padding="same", strides=2),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            Conv2D(128, (3, 3), activation="relu", padding="same", strides=2),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            Conv2D(256, (3, 3), activation="relu", padding="same", strides=2),
            Conv2D(512, (3, 3), activation="relu", padding="same"),
            Conv2D(512, (3, 3), activation="relu", padding="same"),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            Conv2D(256, (1, 1), activation="relu", padding="same"),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            UpSampling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            UpSampling2D((2, 2)),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            Conv2D(2, (3, 3), activation="tanh", padding="same"),
            UpSampling2D((2, 2)),
        ]

    def call(self, x):
        """
        Passes input image through the network.
        """
        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        return tf.keras.losses.MeanSquaredError(
            reduction="auto", name="mean_squared_error"
        )(labels, predictions)
