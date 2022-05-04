import hyperparameters as hp

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

        self.architecture = [
            Dense(units=256, activation="relu", name="dense1_256"),
            Dropout(0.5),
            Dense(units=256, activation="relu", name="dense2_256"),
            Dropout(0.5),
            Dense(units=15, activation="softmax"),
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
