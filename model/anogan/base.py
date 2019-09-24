import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import conv2d, conv2d_transpose, fully_connected, batch_norm

# Custom libs
from utils import TFScalarVariableWrapper

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
                 batch_size,
                 latent_dim,
                 iters_for_encoding):
        """ Initialize AnoGAN object for AnoGAN model

        Args:
            batch_size: Integer, batch size used for test
            latent_dim: Integer, size of latent variable dimension
            iters_for_encoding: Integer, number of iterations to be taken for encoding
        """
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.iters_for_encoding = iters_for_encoding

        with tf.variable_scope('test'):
            # Define latent points to be optimized to find embeddings of given data points
            self.z_trainable = tf.get_variable(name='z_trainable',
                                               shape=[self.batch_size, self.latent_dim],
                                               initializer=tf.truncated_normal_initializer,
                                               trainable=True)

            # Define learning rate 
            self.test_step = tf.get_variable(name='step',
                                             shape=[],
                                             dtype=tf.int64,
                                             initializer=tf.zeros_initializer,
                                             trainable=False)
            learning_rate = tf.train.piecewise_constant(self.test_step,
                                                        boundaries=[200, 300],
                                                        values=[0.01, 0.001, 0.0005])

            # Define optimizer to find latent embedding of given data points 
            self.optimizer_for_encoding = tf.train.AdamOptimizer(learning_rate)

        # Initilizers for variables for test
        self.init_op_test = tf.variables_initializer(tf.global_variables('test'))

    def get_default_session(self):
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError('this method must be called within tf.Session context manager')
        return sess

    def discriminator(self,
                      x_tensor,
                      is_training):
        """ Build discriminator to compute probabilities that given samples are real

        Args:
            x_tensor: Tensor of shape [batch, height, width, channel], a batch of data points
            is_training: Boolean, whether the output of this function will be used for training or inference

        Returns:
            probs: Tensor of shape [batch, 1], probability that given data points are from a true data distribution
            disc_features: Tensor of shape [batch, feature_dim], discriminative feature vectors
        """
        raise NotImplementedError('This function should be implemented.')

    def decoder(self,
                z_tensor,
                is_training):
        """ Generator which maps a latent point to a data point

        Args:
            z_tensor: Tensor of shape [batch, latent_dim], a batch of latent points
            is_training: Boolean, whether the output of this function will be used for training or inference

        Returns:
            output_tensor: Tensor of shape [batch, height, width, channel]
        """
        raise NotImplementedError('This function should be implemented.')

    def build_anomaly_score_function(self,
                                     x_tensor,
                                     labels,
                                     order_norm,
                                     scoring_method,
                                     disc_weight):
        """ Find a latent point z which minimize the residual error between original and reconstructed samples

        Args:
            x_tensor: Tensor of shape [batch, height, width, channel]
            labels: Tensor of shape [batch, 1] or [batch]. Binary labels. 1 is abnormal and 0 is normal.
            order_norm: Integer, used for computing norm.
            scoring_method: String, define how to use discriminative score. \'xent\' for cross entropy or \'fm\' for feature matching error.
            disc_weight: Float, weight for discriminative score in [0,1]. Weight for residual error is defined as (1 - disc_weight).

        Returns:
            anomaly_score: Function which returns anomaly scores
        """
        # Check argument validity 
        assert scoring_method in ['xent', 'fm']
        assert disc_weight >= 0 and disc_weight <= 1

        # Build computational graphs for encoding 
        with tf.name_scope('anomaly_score'):
            x_hat = self.decoder(self.z_trainable, is_training=False)

            # Residual score 
            with tf.name_scope('residual_error'):
                delta = tf.layer.flatten(x_hat - x_tensor)
                residual_error = tf.norm(delta, ord=order_norm, axis=-1, keepdims=False)
                mean_residual_error = tf.reduce_sum(residual_error)

            # Discriminative score 
            y_real, features_real = self.discriminator(x_tensor, is_training=False)
            y_recon, features_recon = self.discriminator(x_hat, is_training=False)
            y_recon = tf.squeeze(y_recon)

            with tf.name_scope('discriminative_error'):
                delta_features = tf.layers.flatten(features_real - features_recon)
                disc_feature_error = tf.norm(delta_features, ord=order_norm, axis=-1, keepdims=False)

            train_for_encoding = self.optimizer_for_encoding.minimize(mean_residual_error,
                                                                      global_step=self.test_step,
                                                                      var_list=[self.z_trainable],
                                                                      name='train_for_encoding')

            disc_score = -y_recon if scoring_method == 'xent' else disc_feature_error
            anomaly_score = (1-disc_weight)*residual_error + disc_weight*disc_score
            labels_squeezed = tf.squeeze(labels)

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

            scores_val, labels_val, x_hat_val = sess.run([anomaly_score, labels_squeezed, x_hat], feed_dict=feed_dict)
            return scores_val, labels_val, x_hat_val

        return anomaly_score_fn

    def build_train_function(self,
                             x_tensor,
                             z_tensor,
                             learning_rate,
                             decay_rate):
        """ Build loss and training graph and create function which performs update on a batch

        Args:
            x_tensor: Tensor of shape [batch, *input_shape]. Batch of input data from true data distribution.
            z_tensor: Tensor of shape [batch, latent_dim]. Batch of latent samples from pre-defined prior distribution.
            learning_rate: Tensor or scala. Learning rate to be used for optimization.
            decay_rate: Tensor or scala. Weight decay rate to be used for optimization.

        Returns:
            train_fn: Function which performs training on a batch and returns loss values
        """
        y_real, features_real = self.discriminator(x_tensor, is_training=True)
        x_fake = self.decoder(z_tensor, is_training=True)
        y_fake, features_fake = self.discriminator(x_fake, is_training=True)

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
            #optimizer['discriminator'] = tf.train.AdamOptimizer(learning_rate=learning_rate/10, beta1=0.5)
            #optimizer['generator'] = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)

        # Define train op
        train_op = {}
        for scope in ['discriminator', 'generator']:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
            with tf.control_dependencies(update_ops):
                train_op[scope] = optimizer[scope].minimize(loss[scope], var_list=tf.trainable_variables(scope))

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

    def create_summary_vars(self):
        """ Create tensors for summary """
        summary_vars = {}
        with tf.variable_scope('summary'):
            summary_vars['discriminator'] = TFScalarVariableWrapper(0, tf.float32, 'discriminator_loss')
            summary_vars['generator'] = TFScalarVariableWrapper(0, tf.float32, 'generator_loss')
        return summary_vars








