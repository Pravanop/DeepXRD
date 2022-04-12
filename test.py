import keras
import train
import numpy as np
import seaborn as sb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class Tester:

    def __init__(self,
                 net: keras.Sequential,
                 data: tuple,
                 ) -> None:
        """

        :param net: keras.sequential object created using models.py
        :param data: dataset in the form of tuples consisting (input, output) pairs
        """
        if net.model_type is 'deepXRD' or 'aCNN':
            self.preds_cnn = np.argmax(net.predict(train.reshape_func(data[0], channels='last')), axis=1)

        if net.model_type is 'seqXRD':
            self.preds_cnn = np.argmax(net.predict(train.reshape_func(data[0], channels='first')), axis=1)

        self.conf_matrix(data[1], self.preds_cnn)

    @staticmethod
    def conf_matrix(true, preds) -> None:
        """
        Plots confusion matrix as a seaborn heatmap
        :param true: true labels
        :param preds: predicted labels
        :return: None
        """
        plt.figure(figsize=(12, 12))
        sb.heatmap(confusion_matrix(true, preds, normalize='true'), annot=True, cmap='viridis',
                   annot_kws={"size": 16})
