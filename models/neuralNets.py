from keras.layers import Conv1D, MaxPool1D, Dense, InputLayer, GlobalAveragePooling1D, Flatten, Dropout, \
    BatchNormalization, LSTM
from keras.models import Sequential
import keras


class neuralNets:
    """
     Definition of deepXRD architecture. Inspired by VGGNet but with batch normalization layers
     and progressively smaller filter sizes and stride lengths.

     Definition of aCNN architecture. Used directly from Oviedo et al.

     Definition of seqXRD architecture. First time use of LSTM for this application.

    """

    def __init__(self,
                 kernel_list=None,
                 stride_list=None,
                 pool_pad: str = 'same',
                 input_shape: int = 1800,
                 model_type: str = 'DeepXRD') -> None:

        """
        :param kernel_list: list of kernel sizes for progressively deeper layers
        :param stride_list: list of stride lengths for progressively deeper layers
        :param pool_pad: type of padding for max pooling layers
        :param input_shape: input shape for cnn model
        """

        self.input_shape = input_shape
        if stride_list is None:
            stride_list = [4, 3, 2, 2, 1]  # Default chosen after experimentation

        if kernel_list is None:
            self.kernel_list = [7, 5, 3, 3, 2]  # Default chosen after experimentation
        self.kernel_list = self.kernel_list
        self.stride_list = stride_list
        self.pool_pad = pool_pad

        if model_type is 'DeepXRD':
            self.model = self.deepxrd()

        if model_type is 'aCNN':
            self.model = self.acnn()

        if model_type is 'seqXRD':
            self.model = self.seqxrd()

    def deepxrd(self) -> keras.Sequential:

        """Definition of deepXRD architecture
        :returns Keras Sequential model"""

        model = Sequential()

        model.add(InputLayer(input_shape=(self.input_shape, 1)))

        model.add(Conv1D(16, self.kernel_list[0], padding='same', activation='relu'))
        model.add(Conv1D(16, self.kernel_list[1], strides=self.stride_list[0], padding='same', activation='relu'))
        model.add(BatchNormalization(0.2))  # TODO change to batch normalization
        model.add(MaxPool1D(2, padding=self.pool_pad))

        model.add(Conv1D(32, self.kernel_list[2], padding='same', activation='relu'))
        model.add(Conv1D(32, self.kernel_list[2], strides=self.stride_list[1], padding='same', activation='relu'))
        model.add(BatchNormalization(0.2))
        model.add(MaxPool1D(2, padding=self.pool_pad))

        model.add(Conv1D(64, self.kernel_list[3], padding='same', activation='relu'))
        model.add(Conv1D(64, self.kernel_list[4], strides=self.stride_list[2], padding='same', activation='relu',
                         name='last_conv_layer'))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(128, activation='relu'))
        model.add(Dense(32))

        model.add(Dense(5, activation='softmax'))

        return model

    def acnn(self) -> keras.Sequential:

        """Immutable architecture directly taken from Oviedo et al.
        :returns Keras sequential object"""

        model = Sequential()

        model.add(Conv1D(32, 8, strides=8, padding='same', input_shape=(self.input_shape, 1), activation='relu'))
        model.add(Conv1D(32, 5, strides=5, padding='same', activation='relu'))
        model.add(Dropout(0.2))
        model.add(Conv1D(32, 3, strides=3, padding='same', activation='relu'))
        model.add(Dropout(0.2))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(5, activation='softmax'))

        return model

    def seqxrd(self) -> keras.Sequential:

        """SeqXRD architecture
        :returns Keras sequential object"""

        model = Sequential()
        input_shape = (1, self.input_shape)
        model.add(LSTM(units=16, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.4))
        model.add(LSTM(units=16, input_shape=input_shape))
        model.add(Dense(units=5, activation='softmax'))

        return model
