import tensorflow as tf, numpy as np

# Import custom libs
from utils import TFObjectWrapper

_SHUFFLE_BUFFER_SIZE = 1000 # Buffer size for shuffling training data

class AnomalyDetectionDataset(TFObjectWrapper):
    """ Wrapper class for tf.data.Dataset for anomaly detection dataset """

    def __init__(self,
                 features,
                 labels,
                 batch_size,
                 is_training):
        """ Initialize a AnomalyDetectionDataset instance
        Args:
            features: Numpy array of shape [nb_data, *input_shape]. Input features.
            labels: Numpy array of shape [nb_data]. Anomaly label
            batch_size: Integer, the number of samples in a batch.
            is_training: Boolean, whether this dataset to be used for training or inference.
        Returns:
            A AnomalyDetectionDataset instance
        """
        # Call super class constructor
        super().__init__()

        # Define attributes
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.is_training = is_training

        # Build tf.data.Dataset from numpy array
        self.dataset = self._build_dataset()

        # Create iterator depending on is_training argument
        if self.is_training:
            self.iterator = self.dataset.make_one_shot_iterator()
        else:
            self.iterator = self.dataset.make_initializable_iterator()

    def initialize(self):
        if self.is_training:
            raise ValueError('When is_training is True, dataset cannot be initialized')
        else:
            sess = self.get_current_session()
            sess.run(self.iterator.initializer)

    def get_next(self):
        """ Override get_next() """
        return self.iterator.get_next()

    def _build_dataset(self):
        """ Build a tf.data.Dataset """

        # Read data using tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((self.features, self.labels))
        if self.is_training:
            dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER_SIZE)
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

