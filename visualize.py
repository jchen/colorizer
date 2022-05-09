from run import *
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb, gray2rgb, rgb2gray
import numpy as np
from tqdm import tqdm

checkpoint = "checkpoint/256/weights.e003.256.h5"
model = build_unet_model((hp.img_size, hp.img_size, 3))
model(keras.Input(shape=(hp.img_size, hp.img_size, 3)))
model.load_weights(checkpoint)

fig, axes = plt.subplots(1, 5, figsize=(25, 5))


def test(image_path, out_dir="output"):
    """
    Tests the model on image at path.

    Visualizes the original image (ground truth), grayscale image, and the predicted image.
    """
    num_images = len(test_images)
    save_path = None

    # fig, axes = plt.subplots(num_images, 5, figsize=(25, 5 * num_images))
    for i, image_path in enumerate(test_images):
        image_rgb = imread(image_path)
        # if image is graycale, convert to RGB
        if len(image_rgb.shape) == 2:
            image_rgb = gray2rgb(image_rgb)

        image_lab = rgb2lab(image_rgb)
        image_l = image_lab[:, :, [0, 0, 0]]
        # Run the model on image_l to get predicted ab channels
        image_ab = model.predict(image_l[np.newaxis, ...])
        image_ab = image_ab[0]
        image_lab[:, :, [1, 2]] = image_ab
        image_rgb_predicted = lab2rgb(image_lab)

        if num_images == 1:
            ax = axes
        else:
            ax = axes[i]

        ax[0].imshow(image_l / 100.0)
        ax[0].set_title("Grayscale Image")

        ax[1].imshow(image_rgb)
        ax[1].set_title("Ground Truth")

        ax[2].imshow(image_rgb_predicted)
        ax[2].set_title("Predicted")

        # Visualize A and B channels
        ax[3].imshow(image_ab[:, :, 0], cmap="gray")
        ax[3].set_title("A Channel")

        ax[4].imshow(image_ab[:, :, 1], cmap="gray")
        ax[4].set_title("B Channel")

        save_path = os.path.join(out_dir, os.path.basename(image_path))

    plt.savefig(save_path)
    # clear(plt)
    # plt.show()


def clear(plt):
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


"""
Usage: python visualize.py
"""
visualize_images = range(1, 36501)
# Sample 100 of visualize_images
visualize_images = np.random.choice(visualize_images, 50, replace=False)
for i in tqdm(visualize_images):
    test_images_numbers = [i]
    test_images = [
        "data/test/val_256/Places365_val_{im:0>8}.jpg".format(im=im)
        for im in test_images_numbers
    ]
    test(test_images)
