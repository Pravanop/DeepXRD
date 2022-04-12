import keras
from keras import optimizers
import numpy as np


class Trainer:
    """
    This model compiles and runs the model
    """

    def __init__(self,
                 net: keras.Sequential,
                 data: tuple,
                 learning_rate: float = 0.001,
                 loss: str = 'sparse_categorical_crossentropy',
                 metrics: object = None,
                 batch_size: int = 128,
                 epochs: int = 100) -> None:

        """

        :param net: keras.sequential object created using models.py
        :param data: dataset in the form of tuples consisting (input, output) pairs
        :param learning_rate: float default taken to be published hyperparamters
        :param loss: default taken to be categorical crossentropy for label encoding
        :param metrics: accuracy and loss as common metrics to be presented while printing
        :param batch_size: integer value taken as 128 as published
        :param epochs: integer value taken as 100 for quick testing
        """

        # TODO add wandb support

        self.data = data

        if metrics is None:
            metrics = ['accuracy', 'loss']
        opt = optimizers.adam_v2.Adam(learning_rate=learning_rate)

        net.compile(optimizer=opt, loss=loss, metrics=metrics)
        if net.model_type is 'deepXRD' or 'aCNN':
            net.fit(reshape_func(data[0], channels='last'),
                    data[1],
                    batch_size=128,
                    epochs=500)
        if net.model_type is 'seqXRD':
            net.fit(reshape_func(data[0], channels='first'),
                    data[1],
                    batch_size=batch_size,
                    epochs=epochs)


def reshape_func(inp: np.array,
                 channels: str = 'first', ) -> np.array:
    """Utility function for adding dimension at required position for CNN or LSTM in keras
    :param channels: first or last as preferred
    :param inp: numpy array
    :returns numpy array with added dimension at preferred position"""

    if channels == 'first':
        inp = inp.reshape(inp.shape[0], 1, inp.shape[1])

    if channels == 'last':
        inp = inp.reshape(inp.shape[0], inp.shape[1], 1)

    return inp
