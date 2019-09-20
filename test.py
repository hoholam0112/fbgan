import os, sys, argparse
from importlib import import_module
from collections import defaultdict

import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

# Custom libs
from utils import TrainConfig, TFScalarVariableWrapper, TFSaverWrapper
from input_pipeline import AnomalyDetectionDataset
from train import get_dataset, get_model

# Define evaluation metric function
def compute_aupr_score(y_trues, y_preds):
    """ Compute AUPR score given y_trues and y_preds

    Args:
        y_trues: 2d-array of shape [batch_size, 1]
        y_preds: 2d-array of shape [batch_size, 1]

    Returns:
        score: float, AUPR score
    """
    precision, recall, thresholds = precision_recall_curve(y_labels, y_preds)
    return auc(recall, precision)

def compute_f1_score(y_trues, y_preds, prevalance):
    """ Compute precision, recall and F1-score

    Args:
        y_trues: 2d-array of shape [batch_size, 1]
        y_preds: 2d-array of shape [batch_size, 1]
        prevalance: float, positive (anomaly) class ratio.

    Returns:
        score: float, AUPR score

    """
    return NotImplementedError('compute_f1_score is called as being not implemented')

def main(args):
    # Load a TrainConfig object from .json file if it exist
    save_dir = os.path.join('./train_logs', args.dataset, args.model, args.save_dir)
    json_path = os.path.join(save_dir, 'train_config.json')

    # Get TrainConfig object from .json file
    train_config = TrainConfig.from_json(json_path)
    print('A TrainConfig object is loaded from a .json file')

    # Get input pipeline
    train_set, test_set = get_dataset(train_config)
    features_train, labels_train = train_set.get_next()
    features_test, labels_test = test_set.get_next()

    # Crate model object 
    model = get_model(train_config)

    # Build a test graph
    anomaly_score_fn = model.build_anomaly_score_function(x_tensor=features_test,
                                                          y_tensor=labels_test,
                                                          order_norm=args.order_norm,
                                                          scoring_method=args.scoring_method,
                                                          disc_weight=args.disc_weight)

    # Define some TF scalar variables to restore and save trainig state
    with tf.variable_scope('train_state'):
        epoch = TFScalarVariableWrapper(0, tf.int64, 'epoch')

    # Create saver
    saver = TFSaverWrapper(train_config.save_dir)

    # Set random seed
    tf.random.set_random_seed(train_config.random_seed)

    with tf.Session() as sess:
        # Load saved model parameters
        saver.restore()
        print('Saved model has been restored')

        # Perform test
        test_set.initialize()
        total_values = defaultdict(lambda: [])
        try:
            max_value = progressbar.UnknownLength
            with progressbar.ProgressBar(max_value) as pbar:
                iters = 0
                while True:
                    # Run the graph
                    scores_val, labels_val, x_hat_val = anomaly_score_fn()

                    # Append graph run results 
                    total_values['score'].append(scores_val)
                    total_values['label'].append(labels_val)
                    total_values['x_hat'].append(x_hat_val)

                    # Update iters
                    iters += 1
                    pbar.update(iters)
        except tf.errors.OutOfRangeError:
            pass

    # Concatenate all values
    for k in total_values.keys():
        total_values[k] = np.concatenate(total_values[k], axis=0)

    # Compute evaluation metrics
    metrics = {}
    metrics['AUROC'] = roc_auc_score(labels_total, scores_total)
    metrics['AUPR'] = compute_aupr_score(labels_total, scores_total)

    for k, v in metrics.items():
        print('{}: {:.4f}'.format(k,v))

if __name__ == '__main__':
    # Parse comand line arguments
    parser = argparse.ArgumentParser('Test a GAN-based anomaly detection model on various datasets')
    parser.add_argument('dataset', help='Dataset name. \'mnist\', \'svhn\' and \'kdd\'.', type=str)
    parser.add_argument('model', help='Name of model. \'anogan\', \'bigan\' and \'fbgan\'.', type=str)
    parser.add_argument('save_dir', help='Name of checkpoint directory', type=str)
    parser.add_argument('--gpu', help='gpu number', type=int)

    # Essential arguments
    parser.add_argument('--scoring_method', help='Scoring method to define anomaly score: \'fm\' or \'xent\'', type=str, default='fm')
    parser.add_argument('--disc_weight', help='Weight for discriminative anomaly score', type=float, default=0)
    parser.add_argument('--order_norm', help='Order of norm to define distance between two compared vectors', type=int, default=1)

    # Parse command arguments
    args = parser.parse_args()

    # Set GPU to be used
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Call main function
    main(args)

