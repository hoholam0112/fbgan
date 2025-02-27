import os, sys, argparse
from importlib import import_module
from collections import defaultdict

import tensorflow as tf, numpy as np
from progressbar import ProgressBar

# Custom libs
from utils import TrainConfig, TFScalarVariableWrapper, TFSaverWrapper
from input_pipeline import AnomalyDetectionDataset
from visualize import grid_plot
from default_params import complete_train_config

def get_train_config(args, json_path):
    """ Create a TrainConfig object from parsed arguments or load it from .json file """
    if os.path.exists(json_path):
        train_config = TrainConfig.from_json(json_path)
        print('A TrainConfig object is loaded from a .json file')
    else:
        train_config = TrainConfig(**args.__dict__)
        print('A TrainConfig object is creatd from parsed arguements')

    # Complete train_config by overiding None arguments by default hyperparameters
    complete_train_config(train_config)
    return train_config

def get_dataset(train_config):
    """ Get AnomalyDetectionDataset objects from train and test set """
    # Import DatasetMarker class and get features and labels  
    dataset_module = import_module('data.{}'.format(train_config.dataset))
    if train_config.dataset == 'kdd':
        dataset_maker = dataset_module.DatasetMaker()
    else:
        dataset_maker = dataset_module.DatasetMaker(train_config.anomaly_labels)
    features_train, labels_train = dataset_maker.get_train_set()
    features_test, labels_test = dataset_maker.get_test_set()

    # Create AnomalyDetectionDataset object from training set 
    train_set = AnomalyDetectionDataset(features=features_train,
                                        labels=labels_train,
                                        batch_size=train_config.batch_size,
                                        is_training=True)

    # Create AnomalyDetectionDataset object from test set 
    test_set = AnomalyDetectionDataset(features=features_test,
                                       labels=labels_test,
                                       batch_size=train_config.batch_size,
                                       is_training=False)
    return train_set, test_set

def get_model(x_tensor_train,
              z_tensor_train,
              train_config):
    """ Create a model instance from given dataset """

    # Import model module
    if train_config.version is not None:
        model_module = import_module('model.{}.{}'.format(train_config.model, train_config.dataset + '_' + train_config.version))
    else:
        model_module = import_module('model.{}.{}'.format(train_config.model, train_config.dataset))

    # Create Model object 
    if train_config.model == 'anogan':
        model = model_module.AnoGAN(x_tensor_train,
                                    z_tensor_train,
                                    batch_size=train_config.batch_size,
                                    latent_dim=train_config.latent_dim,
                                    learning_rate=train_config.learning_rate,
                                    iters_for_encoding=500)
    elif train_config.model == 'bigan':
        model = model_module.BiGAN(x_tensor_train,
                                   z_tensor_train,
                                   batch_size=train_config.batch_size,
                                   latent_dim=train_config.latent_dim,
                                   learning_rate=train_config.learning_rate)
    elif train_config.model == 'fbgan':
        model = model_module.FBGAN(x_tensor_train,
                                   z_tensor_train,
                                   batch_size=train_config.batch_size,
                                   latent_dim=train_config.latent_dim,
                                   learning_rate=train_config.learning_rate,
                                   use_only_fm_loss=train_config.only_fm_loss)
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
    z_random = tf.random.normal(shape=[train_config.batch_size, train_config.latent_dim],
                                name='z_random')

    # Create a summary writer
    logdir = os.path.join(save_dir, 'summary')
    summary_writer = tf.contrib.summary.create_file_writer(logdir, name='summary')
    summary_writer.set_as_default()
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        # Crate model object 
        model = get_model(features_train, z_random, train_config)

        # Define some TF scalar variables to restore and save trainig state
        with tf.variable_scope('train_state'):
            epoch = TFScalarVariableWrapper(0, tf.int64, 'epoch')

        # Create summary ops
        for k, v in model.summary_vars.items():
            tf.contrib.summary.scalar(k + '_summary', v.variable, step=epoch.variable)

        if train_config.dataset in ['mnist', 'svhn']:
            x_fake = model.decoder(z_random, is_training=False)
            tf.contrib.summary.image('generated_images', x_fake, max_images=25, step=epoch.variable)

    # Create saver
    saver = TFSaverWrapper(save_dir, tf.global_variables())

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

        # Initialize summary ops
        tf.contrib.summary.initialize(graph=tf.get_default_graph())

        while epoch.eval() < train_config.end_epoch:
            pbar = ProgressBar()
            mean_loss_val = defaultdict(lambda: 0.0)
            for _ in pbar(range(train_config.iters_per_epoch)):
                # Perform gradient descent on a batch of training data
                loss_val = model.train_on_batch()
                # loss values
                for k, v in loss_val.items():
                    mean_loss_val[k] += v

            # Averaging mean loss values and print it
            print('epoch: {:d}/{:d}'.format(epoch.eval(), train_config.end_epoch), end='>> ')
            for k, v in mean_loss_val.items():
                mean_loss_val[k] = float(v)/train_config.iters_per_epoch
                print('{}: {:.4f}'.format(k + '_loss', mean_loss_val[k]), end=', ')
            print('')

            # Run summary ops
            for key in mean_loss_val.keys():
                model.summary_vars[key].assign(mean_loss_val[key])
            sess.run(tf.contrib.summary.all_summary_ops())

            # Increment epoch
            epoch.assign(epoch.eval() + 1)

            # Save model parameters and training status
            saver.checkpoint()

            # Save TrainConfig object as .json file
            if not os.path.exists(json_path):
                train_config.save(json_path)
                print('A TrainConfig object is saved as a .json file')

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
    parser.add_argument('--version', help='Version of model. Each version has different architectural structure', type=str)

    # Hyperparameters associated with optimization
    parser.add_argument('--batch_size', help='Batch size', type=int)
    parser.add_argument('--learning_rate', help='Learning rate', type=float)
    parser.add_argument('--end_epoch', help='End epoch for training', type=int, default=100)
    parser.add_argument('--iters_per_epoch', help='Number of iterations per one epoch', type=int)
    parser.add_argument('--random_seed', help='Random seed to be used', type=int, default=0)

    # Hyperparameters for particular dataset
    parser.add_argument('--only_fm_loss', help='If this argument is passed, use only fm loss for training encoder and decoder', action='store_true')

    # Parse command arguments
    args = parser.parse_args()

    # Set GPU to be used
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Call main function
    main(args)

