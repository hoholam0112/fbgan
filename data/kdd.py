import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

_RANDOM_SEED = 42
_ANOMALY_RATIO = 0.2 # ratio of abnormal samples
_DATA_PATH = './source/kdd/kddcup.data_10_percent_corrected'

class DatasetMaker:
    """ DatasetMaker class for KDD dataset to use for an anomaly detection task. Anomaly class is 1 (positive), Normal class is 0 (negative). """

    def __init__(self):
        """ Initialize DatasetMaker instance

        Returns:
            dataset_maker: DatasetMaker object whose get_train_set and get_test_set methods return dataset as numpy array.
        """
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

    def _make_dataset(self):
        df = pd.read_csv(_DATA_PATH, header=None, names=self._col_names())
        text_l = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']

        for name in text_l:
            self._encode_text_dummy(df, name)

        labels = df['label'].copy()
        labels[labels != 'normal.'] = 0
        labels[labels == 'normal.'] = 1
        df['label'] = labels

        # split the whole dataset fifty to fifty
        df_train = df.sample(frac=0.5, random_state=_RANDOM_SEED)
        df_test = df.loc[~df.index.isin(df_train.index)]

        x_train, y_train = self._to_xy(df_train, target='label')
        y_train = y_train.flatten().astype(int)
        x_test, y_test = self._to_xy(df_test, target='label')
        y_test = y_test.flatten().astype(int)

        x_train = x_train[y_train != 1]
        y_train = y_train[y_train != 1]

        # Scaling data
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        scaler.transform(x_train)
        scaler.transform(x_test)

        self.features_train = x_train.astype(np.float32)
        self.labels_train = y_train.astype(np.float32)
        self.features_test = x_test.astype(np.float32)
        self.labels_test = y_test.astype(np.float32)

        self._make_testset()

    def _make_testset(self):
        """Adapt the ratio of normal/anomalous data"""
        x_inliers = self.features_test[self.labels_test == 0]
        x_outliers = self.features_test[self.labels_test == 1]

        # Shuffle outlier samples in test set
        indices = self.random_state.permutation(x_outliers.shape[0])
        x_outliers = x_outliers[indices]

        # Get outliers for test set to have a pre-defined anomaly ratio
        rho = _ANOMALY_RATIO
        nb_inliers_test = x_inliers.shape[0]
        nb_outliers_test = int(nb_inliers_test*rho/(1-rho))
        outliers_test = x_outliers[:nb_outliers_test]

        x_test = np.concatenate((x_inliers, outliers_test), axis=0)
        y_test = np.concatenate((np.zeros(nb_inliers_test), np.ones(nb_outliers_test)), axis=0)

        self.features_test, self.labels_test = self._permutate(x_test, y_test)

    def _permutate(self, features, labels):
        """ permutate features and labels """
        perm_idx = self.random_state.permutation(features.shape[0])
        features_perm = features[perm_idx]
        labels_perm = labels[perm_idx]
        return features_perm, labels_perm

    def _col_names(self):
        """Column names of the dataframe"""
        return ["duration","protocol_type","service","flag","src_bytes",
        "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
        "logged_in","num_compromised","root_shell","su_attempted","num_root",
        "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate",
        "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

    def _encode_text_dummy(self, df, name):
        """Encode text values to dummy variables (i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue) """
        dummies = pd.get_dummies(df.loc[:,name])
        for x in dummies.columns:
            dummy_name = "{}-{}".format(name, x)
            df.loc[:, dummy_name] = dummies[x]
        df.drop(name, axis=1, inplace=True)

    def _to_xy(self, df, target):
        """ Converts a Pandas dataframe to the x,y inputs that TensorFlow needs """
        result = []
        for x in df.columns:
            if x != target:
                result.append(x)
        dummies = df[target]
        return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)

