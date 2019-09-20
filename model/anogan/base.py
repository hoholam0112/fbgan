import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import conv2d, conv2d_transpose, fully_connected, batch_norm

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
_TINY = 1e-6 # Tiny positive value for numerical stability for log function

def default_arg_scope(is_training,
                      batch_norm_decay=_BATCH_NORM_DECAY,
                      batch_norm_epsilon=_BATCH_NORM_EPSILON):
    """ arg_scope
    Args:
        is_training: boolean tensor, Whether outputs of the model is used to train or inference
    Returns:
        arg_scope: A arg_scope context manager
    """
    with arg_scope([conv2d, conv2d_transpose, fully_connected],
                   normalizer_fn=batch_norm,
                   normalizer_params={'is_training':is_training,
                                      'center':True,
                                      'scale':True,
                                      'decay':batch_norm_decay,
                                      'epsilon':batch_norm_epsilon,
                                      'fused':True}) as sc:
        return sc

class BaseAnoGAN:
    def __init__(self,
                 latent_dim,
                 iters_for_encoding):
        """ Initialize AnoGAN object for AnoGAN model

        Args:
            latent_dim: Integer, size of latent variable dimension
            iters_for_encoding: Integer, number of iterations to be taken for encoding
        """
        self.latent_dim = latent_dim
        self.iters_for_encoding = iters_for_encoding

        with tf.variable_scope('test_scope'):
            # Define latent points to be optimized to find embeddings of given data points
            self.latent_points = tf.get_variable(name='latent_points',
                                                 shape=[None, self.latent_dim],
                                                 initializer=tf.truncated_normal_initializer,
                                                 trainable=True)

            # Define learning rate 
            self.test_step = tf.get_variable(name='step',
                                             shape=[],
                                             dtype=tf.int64,
                                             initializer=tf.zeros_initializer,
                                             trainable=False)
            learning_rate = tf.train.piecewise_constant(step,
                                                        boundaries=[200, 300],
                                                        values=[0.01, 0.001, 0.0005])

            # Define optimizer to find latent embedding of given data points 
            self.optimizer_for_encoding = tf.train.AdamOptimizer(learning_rate)

        # Initilizers for variables for test
        self.init_op_test = tf.variables_initializer(tf.global_variables('test_scope'))

    def get_default_session(self):
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError('this method must be called within tf.Session context manager')
        return sess

    def discriminator(self,
                      tensor_x,
                      is_training):
        """ Build discriminator to compute probabilities that given samples are real

        Args:
            tensor_x: Tensor of shape [batch, height, width, channel], a batch of data points
            is_training: Boolean, whether the output of this function will be used for training or inference

        Returns:
            probs: Tensor of shape [batch, 1], probability that given data points are from a true data distribution
            disc_features: Tensor of shape [batch, feature_dim], discriminative feature vectors
        """
        raise NotImplementedError('This function should be implemented.')

    def decoder(self,
                tensor_z,
                is_training):
        """ Generator which maps a latent point to a data point

        Args:
            tensor_z: Tensor of shape [batch, latent_dim], a batch of latent points
            is_training: Boolean, whether the output of this function will be used for training or inference

        Returns:
            output_tensor: Tensor of shape [batch, height, width, channel]
        """
        raise NotImplementedError('This function should be implemented.')

    def build_anomaly_score_function(self,
                                     tensor_x,
                                     labels,
                                     order_norm,
                                     scoring_method,
                                     disc_weight):
        """ Find a latent point z which minimize the residual error between original and reconstructed samples

        Args:
            tensor_x: Tensor of shape [batch, height, width, channel]
            labels: Tensor of shape [batch]. Binary labels. 1 is abnormal and 0 is normal.
            order_norm: Integer, used for computing norm.
            scoring_method: String, define how to use discriminative score. \'xent\' for cross entropy or \'fm\' for feature matching error.
            disc_weight: Float, weight for discriminative score in [0,1]. Weight for residual error is defined as 1-disc_weight.

        Returns:
            anomaly_score: Function which returns anomaly scores

        """
        assert scoring_method in ['xent', 'fm']
        assert disc_weight >= 0 and disc_weight <= 1

        # Build computational graphs for encoding 
        with tf.name_scope('anomaly_score'):
            x_hat = self.decoder(self.latent_points, is_training=False)

            # Residual score 
            delta = tf.layer.flatten(x_hat - tensor_x, name='delta')
            residual_error = tf.norm(delta, ord=order_norm, axis=-1, name='residual_error')
            mean_residual_error = tf.reduce_sum(residual_error, name='mean_residual_error')

            # Discriminative score 
            y_real, features_real = self.discriminator(tensor_x, is_training=False)
            y_recon, features_recon = self.discriminator(x_hat, is_training=False)
            y_recon = tf.squeeze(y_recon)
            disc_feature_error = tf.norm(features_real - features_recon, ord=order_norm, axis=-1)

            train_for_encoding = self.optimizer_for_encoding.minimize(mean_residual_error,
                                                                      global_step=self.test_step,
                                                                      var_list=[self.latent_points],
                                                                      name='train_for_encoding')

            disc_score = -y_recon if scoring_method == 'xent' else disc_feature_error
            anomaly_score = (1-disc_weight)*residual_error + disc_weight*disc_score

        def anomaly_score_fn(feed_dict={}):
            """ compute anomaly scores and returns reconstructed images

            Args:
                feed_dict: optional feed_dict argument

            Returns:
                scores_val: numpy 1d-array of type float. Anomaly scores for samples.
                labels_val: numpy 1d-array of type float. Anomaly labels. 1 is abnormal and 0 is normal.
                x_hat_val: numpy 4d-array of shape [batch, height, width, channel]. Reconstruced images.

            """
            # Initailize variables for encoding
            sess = self.get_default_session()
            sess.run(self.init_op_test)

            # Update random latent points to minimize residual error
            for _ in range(self.iters_for_encoding):
                sess.run(train_for_encoding)

            scores_val, labels_val, x_hat_val = sess.run([anomaly_score, labels, x_hat], feed_dict=feed_dict)
            return scores_val, labels_val, x_hat

        return anomaly_score_fn

    def build_train_function(self,
                             tensor_x,
                             tensor_z,
                             learning_rate,
                             decay_rate):
        """ Build loss and training graph and create function which performs update on a batch

        Args:
            tensor_x: Tensor of shape [batch, *input_shape]. Batch of input data from true data distribution.
            tensor_z: Tensor of shape [batch, latent_dim]. Batch of latent samples from pre-defined prior distribution.
            learning_rate: Tensor or scala. Learning rate to be used for optimization.
            decay_rate: Tensor or scala. Weight decay rate to be used for optimization.

        Returns:
            train_fn: Function which performs training on a batch and returns loss values
        """
        y_real = self.discriminator(tensor_x, is_training=True)
        y_fake = self.discriminator(tensor_z, is_training=True)

        # Defined loss function
        with tf.name_scope('loss_functions'):
            loss = {}
            with tf.name_scope('discriminator_loss'):
                loss['discriminator'] = tf.negative(tf.reduce_mean(tf.log(y_real + _TINY) + tf.log(1.0 - y_fake + _TINY)))
            with tf.name_scope('generator_loss'):
                loss['generator'] = tf.negative(tf.reduce_mean(tf.log(y_fake + _TINY)))

        with tf.variable_scope('optimizer_train'):
            optimizer = {}
            optimizer['discriminator'] = tf.contrib.opt.AdamWOptimizer(weight_decay=decay_rate, learning_rate=learning_rate)
            optimizer['generator'] = tf.contrib.opt.AdamWOptimizer(weight_decay=decay_rate, learning_rate=learning_rate)

        # Define train op
        train_op = {}
        for scope in ['discriminator', 'generator']:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                train_op[scope] = optimizer[scope].minimize(loss[scope], var_list=tf.trainable_variable(scope))

        def train_fn():
            """ Train data on a batch

            Args:
                feed_dict: dictionary, optinal feed_dict

            Returns:
                loss_val: Dictionary which maps loss name (string) to loss value (float).
            """
            sess = self.get_default_session()
            loss_val = {}
            for scope in ['discriminator', 'generator']:
                loss_val[scope], _ = sess.run([loss[scope], train_op[scope]])
            return loss_val

        return train_fn

