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
    Rescaling,
)
from tensorflow.keras.applications import VGG16, ResNet50V2


def build_unet_model(input_shape):
    inputs = Input(shape=input_shape)

    vgg16 = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(hp.img_size, hp.img_size, 3),
        input_tensor=inputs,
    )
    for layer in vgg16.layers:
        layer.trainable = False

    resnet = ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=(hp.img_size, hp.img_size, 3),
        input_tensor=inputs,
    )
    for layer in resnet.layers:
        layer.trainable = False

    x = concatenate([vgg16.output, resnet.output])

    # Then passes x through the UNet upsampling model
    x = Conv2DTranspose(256, 3, strides=2, padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = concatenate([x, vgg16.get_layer("block5_conv3").output])

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

        x = concatenate([x, vgg16.get_layer(vgg_layer_name).output])

    x = Conv2DTranspose(64, 3, strides=1, padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(2, 3, strides=1, padding="same")(x)
    x = Activation("sigmoid")(x)
    x = Rescaling(scale=255.0, offset=-128)(x)

    return Model(inputs=inputs, outputs=x)
