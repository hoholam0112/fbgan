import sys, os
import scipy.io
import numpy as np
from subprocess import call

_RANDOM_SEED = 0
_TRAIN_SET_RATIO = 0.8 # Proportion of total normal data to be used as train_set
_TRAIN_PATH = './source/svhn/train_32x32.mat'
_TEST_PATH = './source/svhn/test_32x32.mat'

class DatasetMaker:
    """ DatasetMaker class to transform SVHN dataset for an anomaly detection task. Anomaly class is 1 (positive), Normal class is 0 (negative). """

    def __init__(self,
                 anomaly_labels):
        """ Initialize DatasetMaker instance

        Args:
            anomaly_labels: List of ints. List of anomaly labels. labels which are not in the list is considered normal.

        Returns:
            dataset_maker: DatasetMaker object whose get_train_set and get_test_set methods return dataset as numpy array.
        """

        # Raise AssertError if anomaly_labels is not neither list or int
        assert isinstance(anomaly_labels, (list))
        # Define instance attributes
        self.anomaly_labels = anomaly_labels
        # Define random seed to shuffle original dataset
        self.random_state = np.random.RandomState(seed=_RANDOM_SEED)
        # Build dataset for anomaly detection experiment 
        self._make_dataset()

    def get_train_set(self):
        """ Get training set for anomaly detection.

        Returns:
            features: 4d-numpy array of shape [nb_data, height, width, channel].
            labels: 1d-numpy array of shape [nb_data, 1].
        """
        return self.features_train, self.labels_train

    def get_test_set(self):
        """ Get test set for anomaly detection.

        Returns:
            features: 4d-numpy array of shape [nb_data, height, width, channel].
            labels: 1d-numpy array of shape [nb_data].
        """
        return self.features_test, self.labels_test

    def _download_dataset(self):
        """ Download SVHN dataset if it has not been downloaded yet """
        os.makedirs('./source/svhn', exist_ok=False)
        call('wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat', shell=True, cwd='./source/svhn')
        call('wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat', shell=True, cwd='./source/svhn')

    def _transform_labels(self, original_labels, anomaly_labels):
        """ Transform original labels to binary labels. Anomaly is 1 (positive) and normal is 0 (negative) """
        return np.array([(x in anomaly_labels) for x in original_labels], dtype=np.float32)

    def _permutate(self, features, labels):
        """ permutate features and labels """
        perm_idx = self.random_state.permutation(features.shape[0])
        features_perm = features[perm_idx]
        labels_perm = labels[perm_idx]
        return features_perm, labels_perm

    def _make_dataset(self):
        """ Split dataset for anomaly detection """

        # Download dataset if it has not been downloaded previously.
        if not os.path.isdir('./source/svhn'):
            self._download_dataset()

        # Load dataset
        train_set_origin = scipy.io.loadmat(_TRAIN_PATH)
        test_set_origin = scipy.io.loadmat(_TEST_PATH)

        features_total = np.concatenate([train_set_origin['X'], test_set_origin['X']], axis=-1)
        labels_total = np.concatenate([train_set_origin['y'], test_set_origin['y']], axis=0)

        # Move axis for nb_data is placed at first
        features_total = np.transpose(features_total, [3, 0, 1, 2])

        # Permutate total dataset
        features_total, labels_total = self._permutate(features_total, labels_total)
        labels_total = np.array([(0 if x == 10 else x) for x in np.squeeze(labels_total)], dtype=np.int32)

        # Transform original labels to binary label in [anomaly, normal]. Anomaly is 1 (positive) and normal is 0 (negative).
        labels_trans = self._transform_labels(labels_total, self.anomaly_labels)

        # Seperate anomaly and normal samples
        features_anomaly = features_total[labels_trans.astype(np.bool)]
        features_normal = features_total[(1-labels_trans).astype(np.bool)]

        nb_normal = features_normal.shape[0]
        nb_train = int(nb_normal*_TRAIN_SET_RATIO)
        self.features_train = features_normal[:nb_train]
        self.labels_train = np.zeros(nb_train, dtype=np.float32)

        self.features_test = np.concatenate([features_normal[nb_train:], features_anomaly], axis=0)
        self.labels_test = np.concatenate([np.zeros(nb_normal - nb_train), np.ones(features_anomaly.shape[0])], axis=0)
        self.labels_test = self.labels_test.astype(np.float32)

        # Permutate test_set
        self.features_test, self.labels_test = self._permutate(self.features_test, self.labels_test)

        # Rescale pixel values in [0,1]
        self.features_train = (self.features_train/255.).astype(np.float32)
        self.features_test = (self.features_test/255.).astype(np.float32)

        # Center the pixel values 
        self.features_train = 2*self.features_train - 1
        self.features_test = 2*self.features_test - 1


