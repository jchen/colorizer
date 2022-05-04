import os
import tensorflow as tf

from data import Dataset
import hyperparameters as hp
from model import Model


def train(model, datasets):
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,
    )


def test(model, test_data):
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    dataset = Dataset("data/")
    model = Model()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size)))
    model.summary()


if __name__ == "__main__":
    main()
