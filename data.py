import os
import tensorflow as tf
import hyperparameters as hp
from skimage.color import rgb2lab, gray2rgb


class Dataset:
    """
    Class for containing the training and test sets, as well as other data related functions.
    """

    def __init__(self, data_path):
        """ """
        self.data_path = data_path

        self.train_data = self.get_data(os.path.join(self.data_path, "train/"))

        self.test_data = self.get_data(os.path.join(self.data_path, "test/"))


    def get_data(self, path, shuffle=False, augment=False):
        """
        Gets the data at path, shuffling and augmenting as desired.
        """
        if augment:
            data = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn,
            )
        else:
            data = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn
            )

        img_size = hp.img_size

        data = data.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            batch_size=hp.batch_size,
            shuffle=shuffle,
        )

        return data

    def data_rgb_to_ab(self, data):
        """
        Converts the RGB data to L+AB data.
        """
        for im in data:
            im_lab = rgb2lab(im)
            im_l = im_lab[:, :, [0]]
            im_ab = im_lab[:, :, [1, 2]]
            yield (im_l, im_ab)
