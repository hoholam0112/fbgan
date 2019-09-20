import os, sys, argparse
from importlib import import_module
from collections import defaultdict

import tensorflow as tf, numpy as np
from progressbar import ProgressBar

# Custom libs
from utils import TrainConfig, TFScalarVariableWrapper, TFSaverWrapper
from input_pipeline import AnomalyDetectionDataset
from visualize import grid_plot

def get_train_config(args, json_path):
    """ Create a TrainConfig object from parsed arguments or load it from .json file """
    if os.path.exists(json_path):
        train_config = TrainConfig.from_json(json_path)
        print('A TrainConfig object is loaded from a .json file')
    else:
        train_config = TrainConfig(args)
        print('A TrainConfig object is creatd from parsed arguements')
    return train_config

def get_dataset(train_config):
    """ Get AnomalyDetectionDataset objects from train and test set """
    # Import DatasetMarker class and get features and labels  
    dataset_module = import_module('data.{}'.format(train_config.dataset))
    dataset_maker = dataset_module.DatasetMaker(train_config.anomaly_labels)
    features_train, labels_train = dataset_maker.get_train_set()
    features_test, labels_test = dataset_maker.get_test_set()

    print(features_train.shape)
    print(features_test.shape)
    print(features_train.dtype)
    print(features_test.dtype)
    print(np.max(features_train))
    print(np.max(features_test))
    print(np.min(features_train))
    print(np.min(features_test))
    print(labels_train.shape)
    print(labels_test.shape)
    print(labels_train.dtype)
    print(labels_test.dtype)

    # Create AnomalyDetectionDataset object on train dataset
    train_set = AnomalyDetectionDataset(features=features_train,
                                        labels=labels_train,
                                        batch_size=train_config.batch_size,
                                        is_training=True)
    test_set = AnomalyDetectionDataset(features=features_test,
                                       labels=labels_test,
                                       batch_size=train_config.batch_size,
                                       is_training=False)
    return train_set, test_set

def get_model(train_config):
    """ Import model class and create one depending on dataset """

    # Import model module
    model_module = import_module('{}.{}'.format(train_config.model, train_config.dataset))

    # Create Model object 
    if train_config.model == 'anogan':
        model = model_module.AnoGAN(latent_dim=train_config.latent_dim,
                                    iters_for_encoding=500)
    else:
        raise ValueError('Invalid argument has been passed for train_config.model: {}'.format(train_config.model))
    return model

def main(args):
    """ Train a model on a selected dataset given training configuration """
    # Load a TrainConfig object from .json file if it exist
    save_dir = os.path.join('./train_logs', args.dataset, args.model, args.save_dir)
    json_path = os.path.join(save_dir, 'train_config.json')

    # Create or load a TrainConfig object
    train_config = get_train_config(args, json_path)

    # Get input pipeline
    train_set, test_set = get_dataset(train_config)
    features_train, labels_train = train_set.get_next()
    features_test, labels_test = test_set.get_next()

    with tf.Session() as sess:
        test_set.initialize()
        features_train_val, labels_train_val = sess.run([features_train, labels_train])
        features_test_val, labels_test_val = sess.run([features_test, labels_test])

    indices = np.random.permutation(features_train_val.shape[0])
    images_train = (features_train_val[indices] + 1)/2.0
    y_train = labels_train_val[indices]

    indices = np.random.permutation(features_train_val.shape[0])
    images_test = (features_test_val[indices] + 1)/2.0
    y_test = labels_test_val[indices]

    grid_plot(train_config, images_train, y_train, num_rows=5, num_cols=5, show=False)
    grid_plot(train_config, images_test, y_test, num_rows=5, num_cols=5, show=True)

    sys.exit()

    # Crate model object 
    model = get_model(train_config)

    # Build a training graph
    train_fn = model.build_train_function(x_tensor=features_train,
                                          z_tensor=tf.random.truncated_normal(shape=[train_config.batch_size, train_config.latent_dim]))

    # Define some TF scalar variables to restore and save trainig state
    with tf.variable_scope('train_state'):
        epoch = TFScalarVariableWrapper(0, tf.int64, 'epoch')

    # Create saver
    saver = TFSaverWrapper(train_config.save_dir)

    # Define an initialization op for global TF variables
    glob_init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Initialize or restore training state
        if os.path.isdir(save_dir):
            saver.restore()
            print('Lastest training state restored')
        else:
            # Initialize all TF variables
            sess.run(glob_init_op)
            print('Initialized global variables')

        while epoch.eval() < train_config.end_epoch:
            pbar = ProgressBar()
            mean_loss_val = defaultdict(lambda: 0.0)
            for _ in pbar(range(train_config.iters_per_epoch)):
                # Perform gradient descent on a batch of training data
                loss_val = train_fn()

                # loss values
                for k, v in loss_val.items():
                    mean_loss_val[k] += v

            # Averaging mean loss values and print it
            for k, v in mean_loss_val.items():
                mean_loss_val[k] = v/float(train_config.iters_per_epoch)
                print('{}: {.4f}'.format(k, mean_loss_val[k]), end=' | ')
            print('')

if __name__ == '__main__':
    # Parse comand line arguments
    parser = argparse.ArgumentParser('Train a GAN-based anomaly detection model on various datasets')
    parser.add_argument('dataset', help='Dataset name. \'mnist\', \'svhn\' and \'kdd\'.', type=str)
    parser.add_argument('model', help='Name of model. \'anogan\', \'bigan\' and \'fbgan\'.', type=str)
    parser.add_argument('save_dir', help='Name of checkpoint directory', type=str)
    parser.add_argument('--gpu', help='gpu number', type=int)

    # Essential arguments
    parser.add_argument('--anomaly_labels', help='List of anomaly labels', type=int, nargs='+', required=True)
    parser.add_argument('--latent_dim', help='latent dimension size', type=int)

    # Hyperparameters associated with optimization
    parser.add_argument('--batch_size', help='Batch size', type=int, default=100)
    parser.add_argument('--end_epoch', help='End epoch for training', type=int)
    parser.add_argument('--iters_per_epoch', help='Number of iterations per one epoch', type=int, default=300)
    parser.add_argument('--random_seed', help='Random seed to be used', type=int, default=0)

    # Parse command arguments
    args = parser.parse_args()

    # Set GPU to be used
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Call main function
    main(args)




