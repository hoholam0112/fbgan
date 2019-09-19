import numpy as np
import matplotlib.pyplot as plt

class dataset:
    def __init__(self, label, label_is_anomaly=true, centered=true, flatten=false):
        self.random_state = np.random.randomstate(0)
        self.path = '/home/mlg/ynk/public_data'
        self.ratio = 0.2
        self.label = label
        self.centered = centered
        self.flatten = flatten
        self.label_is_anomaly = label_is_anomaly
        self.generate_dataset()

    def get_data_spec(self):
        """
            get a dictionary of the data specification
            input_shape, nb_class
        """
        return {'is_image' : true, 'sample_shape' : [28, 28, 1], 'anomaly_label' : 1}

    def get_train_set(self):
        return self.x_train.astype(np.float32)

    def get_test_set(self):
        """
			abnormal class is denoted as 1, and normal classes are denoted as 0
        """
        return self.x_test.astype(np.float32), self.y_test.astype(np.int32)

    def generate_dataset(self):
        """gets the adapted dataset for the experiments

        args :
            anomalous_class (int): int in range 0 to 10, is the class/digit
                         which is considered outlier
            centered (bool): (default=false) data centered to [-1, 1]
            flatten (bool): (default=false) flatten the data
        returns :
                (tuple): <training, testing> images and labels
        """

        data = np.load('{}/mnist.npz'.format(self.path))
        # print(data.keys())

        if self.label_is_anomaly:
            full_x_data = np.concatenate([data['x_train'], data['x_test'], data['x_valid']], axis=0)
            full_y_data = np.concatenate([data['y_train'], data['y_test'], data['y_valid']], axis=0)

            perm_idx = self.random_state.permutation(full_x_data.shape[0])
            full_x_data = full_x_data[perm_idx]
            full_y_data = full_y_data[perm_idx]

            x_normal = full_x_data[full_y_data != self.label]
            y_normal = full_y_data[full_y_data != self.label]
            x_anomaly = full_x_data[full_y_data == self.label]
            y_anomaly = full_y_data[full_y_data == self.label]

            nb_train_data = int(x_normal.shape[0]*(1-self.ratio))
            self.x_train = x_normal[:nb_train_data]
            x_normal_test = x_normal[nb_train_data:]

            self.x_test = np.concatenate([x_normal_test, x_anomaly], axis=0)
            self.y_test = np.concatenate([np.zeros(x_normal_test.shape[0]), np.ones(x_anomaly.shape[0])], axis=0)

        else:
            x_train = np.concatenate([data['x_train'], data['x_valid']], axis=0)
            y_train = np.concatenate([data['y_train'], data['y_valid']], axis=0)
            self.x_test = data['x_test']
            self.x_train = (x_train[y_train == self.label])
            self.y_train = np.zeros(self.x_train.shape[0])

            self.y_test = np.zeros(self.x_test.shape[0])
            self.y_test[data['y_test'] == self.label] = 0
            self.y_test[data['y_test'] != self.label] = 1

        if self.centered:
            self.x_train = self.x_train*2-1
            self.x_test = self.x_test*2-1

        if not self.flatten:
            self.x_train = np.reshape(self.x_train, [self.x_train.shape[0], 28, 28, 1])
            self.x_test = np.reshape(self.x_test, [self.x_test.shape[0], 28, 28, 1])

        return none

if __name__ == '__main__':
    dataset = dataset(label=2, label_is_anomaly=false)
    x_train = dataset.get_train_set()
    x_test, y_test = dataset.get_test_set()

    print(x_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    perm_idx = np.random.permutation(x_train.shape[0])
    x_train = x_train[perm_idx]

    perm_idx = np.random.permutation(x_test.shape[0])
    x_test = x_test[perm_idx]
    y_test = y_test[perm_idx]

    images = x_test[:25]
    plt.figure()
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.imshow(images[i, :, :, 0], cmap='gray')
    plt.show()
