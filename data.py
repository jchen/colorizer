import os
import tensorflow as tf
import hyperparameters as hp
from skimage.color import rgb2lab, gray2rgb
import numpy as np
import cv2


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

        return self.data_rgb_to_l_ab(data)

    
    # def data_rgb_to_l_ab(self, data):
    #     """
    #     Converts the RGB data to L+AB data.
    #     """
    #     for im in data:
    #         im_lab = rgb2lab(im)
    #         im_l = im_lab[:, :, [0]]
    #         im_ab = im_lab[:, :, [1, 2]]
    #         yield (im_l, im_ab)
    #     return im_l
            
            
    # Convert all training images from the RGB color space to the Lab color space.
    # def rgb_to_lab(self, train_imgs):
    #     X = []
    #     y = []
    #     for img in train_imgs[0]:
    #         lab = rgb2lab(img)
    #         X.append(lab[:,:,0])
    #         y.append(lab[:,:,1:] / 128)  
    #     X = np.array(X)
    #     y = np.array(y)

    #     X = X.reshape(X.shape+(1,))
            
    # Use the L channel as the input to the network and train the network to predict the ab channels.
    def predict_ab(self, L):
        print("Predicting AB channels")
        
    
    # load and scale image
    def preprocess(self, image):
        # image = cv2.imread(image)
        scaled = image.astype("float32") / 255.0
        resized = cv2.resize(scaled, (224, 224))
        
        # # lab postprocess
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        resized = cv2.resize(lab, (224, 224)) # dimensions the colorization network accepts
        L = cv2.split(resized)[0]
        L -= 50 # mean centering