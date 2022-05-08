import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import hyperparameters as hp
from skimage.color import rgb2lab, gray2rgb


class Dataset:
    """
    Class for containing the training and test sets, as well as other data related functions.
    """

    def __init__(self, data_path):
        """
        Initialize the Dataset object.
        """
        self.data_path = data_path

        self.train_data = self.get_data(os.path.join(self.data_path, "train/"))

        self.test_data = self.get_data(os.path.join(self.data_path, "test/"))

    def get_data(self, path, shuffle=False, augment=False):
        """
        Gets the data at path, shuffling and augmenting as desired.
        """
        if augment:
            data = ImageDataGenerator(
                preprocessing_function=self.preprocess_fn,
            )
        else:
            data = ImageDataGenerator(preprocessing_function=self.preprocess_fn)

        img_size = hp.img_size

        data = data.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            batch_size=hp.batch_size,
            shuffle=shuffle,
        )

        return self.data_rgb_to_l_ab(data)

    def data_rgb_to_l_ab(self, data):
        """
        Converts the RGB data to L+AB data.
        """
        img_size = hp.img_size
        for im in data:
            im = im[0]
            im_lab = rgb2lab(im)
            # We need 3 identical channels because of our pretrained model backbones.
            im_l = im_lab[:, :, :, [0, 0, 0]]
            im_ab = im_lab[:, :, :, [1, 2]]
            yield (im_l, im_ab)

    def standardize(self, img):
        return img

    def preprocess_fn(self, img):
        """Preprocess function for ImageDataGenerator."""
        img = img / 255.0
        img = self.standardize(img)
        return img
