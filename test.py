import os, sys, argparse
from importlib import import_module
from collections import defaultdict

import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
import progressbar
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, \
        precision_recall_fscore_support

# Custom libs
from utils import TrainConfig, TFScalarVariableWrapper, TFSaverWrapper
from input_pipeline import AnomalyDetectionDataset
from train import get_dataset, get_model
from visualize import grid_plot

# Define evaluation metric function
def compute_aupr_score(y_trues, y_preds):
    """ Compute AUPR score given y_trues and y_preds

    Args:
        y_trues: 2d-array of shape [batch_size, 1]
        y_preds: 2d-array of shape [batch_size, 1]

    Returns:
        score: float, AUPR score
    """
    precision, recall, thresholds = precision_recall_curve(y_trues, y_preds)
    return auc(recall, precision)

def compute_f1_score(y_trues, y_preds, prevalance):
    """ Compute precision, recall and F1-score

    Args:
        y_trues: 2d-array of shape [batch_size, 1]
        y_preds: 2d-array of shape [batch_size, 1]
        prevalance: float, positive (anomaly) class ratio.

    Returns:
        precision: float, precision
        recall: float, recall
        f1_score: float, F1-score, harmornic mean of precision and recall
    """
    assert prevalance >= 0 and prevalance <= 1

    cut = np.percentile(y_preds, (1.0 - prevalance)*100)
    y_preds_bin = [(1 if x >= cut else 0) for x in y_preds]

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_trues, y_preds_bin, average='binary')
    return precision, recall, f1_score

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
    z_random = tf.random.normal(shape=[train_config.batch_size, train_config.latent_dim],
                                dtype=tf.float32,
                                name='z_random')

    # Crate model object 
    model = get_model(features_train, z_random, train_config)

    # Build a test graph
    anomaly_score_fn = model.build_anomaly_score_function(x_tensor=features_test,
                                                          y_tensor=labels_test,
                                                          order_norm=args.order_norm,
                                                          scoring_method=args.scoring_method,
                                                          disc_weight=args.disc_weight)

    x_fake = model.decoder(z_random, is_training=False)

    # Create saver
    saver = TFSaverWrapper(save_dir, var_list=model.vars_to_restore)

    # Set random seed
    tf.random.set_random_seed(train_config.random_seed)

    with tf.Session() as sess:
        # Load saved model parameters
        saver.restore()
        print('Saved model has been restored')

        # Perform test
        test_set.initialize()
        output_values_total = defaultdict(lambda: [])
        try:
            max_value = progressbar.UnknownLength
            with progressbar.ProgressBar(max_value) as pbar:
                iters = 0
                while True:
                    # Run the graph
                    output_values = anomaly_score_fn()

                    # Append graph run results 
                    for k, v in output_values.items():
                        output_values_total[k].append(v)

                    # Update iters
                    iters += 1
                    pbar.update(iters)
        except tf.errors.OutOfRangeError:
            pass

        x_fake_val = sess.run(x_fake)

    # Concatenate all values
    for k, v in output_values_total.items():
        output_values_total[k] = np.concatenate(v, axis=0)

    # Compute evaluation metrics
    metrics = {}
    if train_config.dataset in ['kdd']:
        precision, recall, f1_score = compute_f1_score(output_values_total['label'], output_values_total['score'], 0.2)
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1_score
    else:
        metrics['AUROC'] = roc_auc_score(output_values_total['label'], output_values_total['score'])
        metrics['AUPR'] = compute_aupr_score(output_values_total['label'], output_values_total['score'])

    # Show performance of a model
    for k, v in metrics.items():
        print('{}: {:.4f}'.format(k,v))

    if train_config.dataset in ['svhn', 'mnist']:
        # Show qualitative results
        indices_rand = np.random.permutation(output_values_total['x_real'].shape[0])
        images_shuffled = output_values_total['x_real'][indices_rand]
        reconstructs_shuffled = output_values_total['x_hat'][indices_rand]
        labels_shuffled = output_values_total['label'][indices_rand]

        grid_plot(train_config, images_shuffled, labels_shuffled, num_rows=5, num_cols=5, show=False)
        grid_plot(train_config, x_fake_val, np.zeros([x_fake_val.shape[0]]), num_rows=5, num_cols=5, show=False)
        grid_plot(train_config, reconstructs_shuffled, labels_shuffled, num_rows=5, num_cols=5, show=True)

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

