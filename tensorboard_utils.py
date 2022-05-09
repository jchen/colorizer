"""
Originally for...
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University

Edited by TNMT for final project. 
"""

import io
import os
import re
import sklearn.metrics
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import hyperparameters as hp


class CustomModelSaver(tf.keras.callbacks.Callback):
    """Custom Keras callback for saving weights of networks."""

    def __init__(self, checkpoint_dir, max_num_weights=1000):
        super(CustomModelSaver, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.max_num_weights = max_num_weights

    def on_epoch_end(self, epoch, logs=None):
        """At epoch end, weights are saved to checkpoint directory."""
        cur_acc = logs["val_mean_squared_error"]
        save_name = "weights.e{0:03d}.{1:03d}.h5".format(epoch, hp.img_size)
        self.model.save_weights(self.checkpoint_dir + os.sep + save_name)
