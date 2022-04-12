import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import List


class DataFlow:
    """
    Given the dataset and source, target preferences two tuples corresponding to train and test data are outputted
    """

    def __init__(self,
                 data_dict: dict,
                 source: str = 'all',
                 target_type: str = 'label',
                 augmentation: bool = True,
                 split: float = 0.1) -> None:

        """

        :param data_dict: the dataset in the form of dict obtained from data scraper
        :param source: source can be 'all' or 'Cu', 'Fe' etc.
        :param target_type: label or one hot encoder target. takes in 'label' or 'ohec'
        :param augmentation: bool for augmentation of dataset with smote or not. True for smote augmentation
        """

        self.split = split
        self.target_type = target_type
        self.data = data_dict
        self.count = self.target_counter()
        self.source = source
        self.selected_targets = []
        self.target_select()

        self.inputs, self.targets_unlabelled = self.datagen()

        if target_type == 'label':
            self.targets = self.label_encoder(self.targets_unlabelled)

        if self.target_type == 'onehot':
            self.targets = self.onehot_encoder(self.targets_unlabelled)

        if augmentation:
            self.train, self.test = self.smote_aug()

        else:
            self.train, self.test = self.without_smote_aug()

    def target_counter(self) -> dict:

        """Gets the count of each space group in the dataset
        :returns dictionary of counter object"""

        targets = []
        for i in self.data.keys():

            for j in range(4):
                targets.append(self.data[i]['Y']['spacegroup'])

        return dict(Counter(targets))

    def target_select(self) -> None:
        """Appends only space group above a certain count to avoid data sparsity.
        Here chosen to be 400 samples"""

        for i in self.count.keys():
            if self.count[i] >= 400:
                self.selected_targets.append(i)

    def datagen(self) -> (np.array, list):

        """For the selected targets and a specified source the dataset is ordered as inputs and targets

        :returns X as numpy array and Y as list for further processing
        """

        X = []
        Y = []

        for key in self.data.keys():

            if self.data[key]['X'] is None:
                continue

            else:

                if self.data[key]['Y']['spacegroup'] in self.selected_targets:

                    xrd_dict = self.data[key]['X']

                    if self.source == 'all':

                        for j in xrd_dict.keys():
                            X.append(xrd_dict[j])
                            Y = [self.data[key]['Y']['spacegroup'] for _ in range(4)]

                    else:
                        X.append(xrd_dict['xrd.' + self.source])
                        Y.append(self.data[key]['Y']['spacegroup'])

        assert len(X) == len(Y)
        print("No. of samples: ", len(X))

        return np.array(X), Y

    @staticmethod
    def label_encoder(target_names: List[str]) -> np.array:

        """Label encoder for targets_list
        :parameter target_names: list of string targets
        :returns numpy array of labels"""

        lenc = LabelEncoder()
        return np.array(lenc.fit_transform(target_names))

    @staticmethod
    def onehot_encoder(target_names: List[str]) -> np.array:
        """One hot Encoder for targets_list
        :parameter target_names: list of string targets
        :returns numpy array of labels"""

        ohec = OneHotEncoder()
        return ohec.fit_transform(np.array(target_names).reshape(-1, 1))

    def without_smote_aug(self) -> ((np.array, np.array), (np.array, np.array)):

        """For a specified target encoding and dataset split, shuffled train and test sets are created.

        Note: Use this only if you don't want smote augmentation!

        :returns Two tuples containing training inputs and training outputs, and
                 testing inputs and testing outputs
        """

        X_train, X_test, Y_train, Y_test = train_test_split(self.inputs, self.targets, test_size=self.split,
                                                            random_state=42)

        return (X_train, Y_train), (X_test, Y_test)

    def smote_aug(self) -> ((np.array, np.array), (np.array, np.array)):

        """
        For a specified target encoding, first smote augmentation is done
        and for a dataset split, shuffled train and test sets are created.

         :returns Two tuples containing training inputs and training outputs, and
                         testing inputs and testing outputs
        """

        oversample = SMOTE()

        X_smote, Y_smote = oversample.fit_resample(self.inputs, self.targets)

        X_train, X_test, Y_train, Y_test = train_test_split(X_smote, Y_smote, test_size=self.split,
                                                            random_state=42)

        return (X_train, Y_train), (X_test, Y_test)
