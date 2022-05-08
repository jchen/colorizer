import os
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()
from tensorflow import keras

from data import Dataset
import hyperparameters as hp

# from model import UNetModel as Model
from model import build_unet_model

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
    model = build_unet_model((hp.img_size, hp.img_size, 3))
    model(keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    print(model.summary())
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.learning_rate),
        loss="mean_squared_error",
        metrics=[keras.metrics.MeanSquaredError()],
    )
    train(model, dataset)


if __name__ == "__main__":
    main()
