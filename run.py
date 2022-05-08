import os
import tensorflow as tf

from tensorflow import keras

from data import Dataset
import hyperparameters as hp
from model import UNetModel as Model

from tensorboard_utils import CustomModelSaver

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def train(model, datasets, logs_path="logs/", checkpoint_path="checkpoint/"):

    callback_list = [
        keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq="batch",
            profile_batch=0,
        ),
        CustomModelSaver(checkpoint_path, hp.max_num_weights),
    ]

    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        steps_per_epoch=hp.steps_per_epoch,
        validation_steps=hp.validation_steps,
        callbacks=callback_list,
    )


def test(model, test_data):
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    dataset = Dataset("data/")
    model = Model()
    model(keras.Input(shape=(hp.img_size, hp.img_size, 1)))
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=[keras.metrics.MeanSquaredError],
    )
    print(model.summary())
    train(model, dataset)


if __name__ == "__main__":
    main()
