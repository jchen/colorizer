import hyperparameters as hp

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense


class Model(tf.keras.model):
    def __init__(self):
        super(Model, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)

    @staticmethod
    def loss_fn(labels, predictions):
        pass
