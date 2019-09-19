import os, sys, argparse
from importlib import import_module

import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt

# Custom libs
from utils import TrainConfig
from data.dataset import AnomalyDetectionDataset

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
    features_train, labels_train= dataset_maker.get_train_set()
    features_test, labels_test = dataset_maker.get_test_set()

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

def train(save_dir,
          train_config,
          train_set,
          valid_set,
          model,
          loss_fn,
          metric_fns,
          optimizer):
    return

def main(args):
    """ Train a model on a selected dataset given training configuration """

    # Load a TrainConfig object from .json file if it exist
    save_dir = os.path.join('./train_logs', args.dataset, args.model, args.save_dir)
    json_path = os.path.join(save_dir, 'train_config.json')

    # Create or load a TrainConfig object
    train_config = get_train_config(args, json_path)

    train_set, test_set = get_dataset(train_config)

    features_train, labels_train = train_set.get_next()
    features_test, labels_test = test_set.get_next()

    with tf.Session() as sess:
        test_set.initialize()
        features_train_val, labels_train_val = sess.run([features_train, labels_train])
        features_test_val, labels_test_val = sess.run([features_test, labels_test])


    return




if __name__ == '__main__':
    # Parse comand line arguments
    parser = argparse.ArgumentParser('Train a GAN-based anomaly detection model on various datasets')
    parser.add_argument('dataset', help='Dataset name. \'mnist\', \'svhn\' and \'kdd\'.', type=str)
    parser.add_argument('model', help='Name of model. \'anogan\', \'bigan\' and \'fbgan\'.', type=str)
    parser.add_argument('save_dir', help='Name of checkpoint directory', type=str)
    parser.add_argument('--gpu', help='gpu number', type=int)

    # Essential arguments
    parser.add_argument('--anomaly_labels', help='List of anomaly labels', type=int, nargs='+', required=True)

    # Hyperparameters associated with optimization
    parser.add_argument('--batch_size', help='Batch size', type=int, default=100)

    # Parse command arguments
    args = parser.parse_args()

    # Set GPU to be used
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Call main function
    main(args)




