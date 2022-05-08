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

    def __init__(self, checkpoint_dir, max_num_weights=5):
        super(CustomModelSaver, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.max_num_weights = max_num_weights

    def on_epoch_end(self, epoch, logs=None):
        """At epoch end, weights are saved to checkpoint directory."""

        min_acc_file, max_acc_file, max_acc, num_weights = self.scan_weight_files()

        cur_acc = logs["val_mean_squared_error"]

        # Only save weights if test accuracy exceeds the previous best
        # weight file
        if cur_acc > max_acc:
            save_name = "weights.e{0:03d}-acc{1:.4f}.h5".format(epoch, cur_acc)

            self.model.save_weights(self.checkpoint_dir + os.sep + "your." + save_name)

            # Ensure max_num_weights is not exceeded by removing
            # minimum weight
            if self.max_num_weights > 0 and num_weights + 1 > self.max_num_weights:
                os.remove(self.checkpoint_dir + os.sep + min_acc_file)

    def scan_weight_files(self):
        """Scans checkpoint directory to find current minimum and maximum
        accuracy weights files as well as the number of weights."""

        min_acc = float("inf")
        max_acc = 0
        min_acc_file = ""
        max_acc_file = ""
        num_weights = 0

        files = os.listdir(self.checkpoint_dir)

        for weight_file in files:
            if weight_file.endswith(".h5"):
                num_weights += 1
                file_acc = float(
                    re.findall(r"[+-]?\d+\.\d+", weight_file.split("acc")[-1])[0]
                )
                if file_acc > max_acc:
                    max_acc = file_acc
                    max_acc_file = weight_file
                if file_acc < min_acc:
                    min_acc = file_acc
                    min_acc_file = weight_file

        return min_acc_file, max_acc_file, max_acc, num_weights
