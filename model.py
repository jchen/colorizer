import hyperparameters as hp

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    concatenate,
    Conv2D,
    Conv2DTranspose,
    MaxPool2D,
    Dropout,
    Flatten,
    Dense,
    UpSampling2D,
    Reshape,
)
from tensorflow.keras.applications import VGG16, ResNet50V2


class UpsamplingModel(Model):
    def __init__(self):
        super(UpsamplingModel, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        # self.optimizer = tf.keras.optimizers.SGD(learning_rate=hp.learning_rate)

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

    def summary(self):
        x = Input(shape=(256, 256, 1))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()


class UNetModel(Model):
    def __init__(self):
        super(UNetModel, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
        self.vgg_model = VGG16(
            include_top=False,
            weights="imagenet",
            input_shape=(hp.img_size, hp.img_size, 3),
        )
        for layer in self.vgg_model.layers:
            layer.trainable = False

        self.resnet_model = ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_shape=(hp.img_size, hp.img_size, 3),
        )

    def call(self, x):
        """
        Passes input image through the network.
        """
        # First copies x into 3-channel input
        x = tf.tile(x, [1, 1, 1, 3])
        # Then passes x through the VGG16 model
        x1 = self.vgg_model(x)
        x2 = self.resnet_model(x)
        x = concatenate([x1, x2])
        # Then passes x through the UNet upsampling model
        x = self.upsampling_model(x)

        return x

    def upsampling_model(self, x):
        """
        Expects the output of VGG16, and constructs the upsampling end of a UNet model.
        """
        x = Conv2DTranspose(256, 3, strides=2, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)

        x = concatenate([x, self.vgg_model.get_layer("block5_conv3").output])

        for filter_size, vgg_layer_name in [
            (512, "block4_conv3"),
            (512, "block3_conv3"),
            (256, "block2_conv2"),
            (128, "block1_conv2"),
        ]:
            # Deconvolution to match VGG layers, then concatenate with VGG layer output
            x = Conv2DTranspose(filter_size, 3, strides=1, padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization()(x)

            x = Conv2DTranspose(filter_size, 3, strides=2, padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization()(x)

            x = concatenate([x, self.vgg_model.get_layer(vgg_layer_name).output])

        x = Conv2DTranspose(64, 3, strides=1, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(2, 3, strides=2, padding="same")(x)
        x = BatchNormalization()(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions):
        return tf.keras.losses.MeanSquaredError(
            reduction="auto", name="mean_squared_error"
        )(labels, predictions)

    def summary(self):
        x = Input(shape=(256, 256, 1))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()
