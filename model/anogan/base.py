import tensorflow as tf, numpy as np
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import conv2d, conv2d_transpose, fully_connected, batch_norm

# Custom libs
from utils import TFScalarVariableWrapper

_TINY = 1e-6 # Tiny positive value for numerical stability for log function

def default_arg_scope(is_training, **kwargs):
    """ arg_scope
    Args:
        is_training: boolean tensor, Whether outputs of the model is used to train or inference
        kwargs: keyword arguments
    Returns:
        arg_scope: A arg_scope context manager
    """
    default_kwargs = dict(normalizer_fn=batch_norm,
                          normalizer_params={'is_training':is_training,
                                             'center':True,
                                             'scale':True,
                                             'fused':True},
                          weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
    default_kwargs.update(kwargs)
    with arg_scope([conv2d, conv2d_transpose, fully_connected],
                   **default_kwargs) as sc:
        return sc

class BaseAnoGAN:
    def __init__(self,
                 x_tensor_train,
                 z_tensor_train,
                 batch_size,
                 latent_dim,
                 learning_rate,
                 iters_for_encoding):
        """ Initialize AnoGAN object for AnoGAN model

        Args:
            x_tensor_train: Tensor of shape [batch, *input_shape]. Batch of input data from true data distribution.
            z_tensor_train: Tensor of shape [batch, latent_dim]. Batch of latent samples from pre-defined prior distribution.
            batch_size: Integer, batch size used for test
            latent_dim: Integer, size of latent variable dimension
            learning_rate: Tensor or scala. Learning rate to be used for optimization.
            iters_for_encoding: Integer, number of iterations to be taken for encoding
        """
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.iters_for_encoding = iters_for_encoding

        # Build a training graph
        y_real, features_real = self.discriminator(x_tensor_train, is_training=True)
        x_fake = self.decoder(z_tensor_train, is_training=True)
        y_fake, features_fake = self.discriminator(x_fake, is_training=True)

        # Defined loss function
        with tf.name_scope('loss_functions'):
            self.loss = {}
            with tf.name_scope('discriminator_loss'):
                self.loss['discriminator'] = tf.negative(tf.reduce_mean(tf.log(y_real + _TINY) + tf.log(1.0 - y_fake + _TINY)))
            with tf.name_scope('generator_loss'):
                self.loss['generator'] = tf.negative(tf.reduce_mean(tf.log(y_fake + _TINY)))

        # Define train ops
        with tf.name_scope('train_ops'):
            self.train_op = {}
            self.ema = tf.train.ExponentialMovingAverage(decay=0.999)
            for scope in ['discriminator', 'generator']:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
                with tf.control_dependencies(update_ops):
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='Adam_{}'.format(scope))
                    self.train_op[scope] = optimizer.minimize(self.loss[scope], var_list=tf.trainable_variables(scope))

                # Create ops for exponential moving average 
                with tf.control_dependencies([self.train_op[scope]]):
                    self.train_op[scope] = tf.group(self.ema.apply(tf.trainable_variables(scope)))

        # Creates summary vars
        self.summary_vars = {}
        with tf.variable_scope('summary'):
            self.summary_vars['discriminator'] = TFScalarVariableWrapper(0, tf.float32, 'discriminator_loss')
            self.summary_vars['generator'] = TFScalarVariableWrapper(0, tf.float32, 'generator_loss')

        # Vars to restore
        #self.vars_to_restore = self.ema.variables_to_restore()
        self.vars_to_restore = tf.global_variables()

    def get_default_session(self):
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError('this method must be called within tf.Session context manager')
        return sess

    def train_on_batch(self):
        """ Train data on a batch """
        sess = self.get_default_session()
        loss_val = {}
        for scope in ['discriminator', 'generator']:
            loss_val[scope], _ = sess.run([self.loss[scope], self.train_op[scope]])
        return loss_val

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
                                     y_tensor,
                                     order_norm,
                                     scoring_method,
                                     disc_weight):
        """ Find a latent point z which minimize the residual error between original and reconstructed samples

        Args:
            x_tensor: Tensor of shape [batch, height, width, channel]
            y_tensor: Tensor of shape [batch, 1] or [batch]. Binary y_tensor. 1 is abnormal and 0 is normal.
            order_norm: Integer, used for computing norm.
            scoring_method: String, define how to use discriminative score. \'xent\' for cross entropy or \'fm\' for feature matching error.
            disc_weight: Float, weight for discriminative score in [0,1]. Weight for residual error is defined as (1 - disc_weight).

        Returns:
            anomaly_score: Function which returns anomaly scores
        """
        # Check argument validity 
        assert scoring_method in ['xent', 'fm']
        assert disc_weight >= 0 and disc_weight <= 1

        # Define latent points to be optimized to find embeddings of given data points
        with tf.variable_scope('test'):
            z_trainable = tf.get_variable(name='z_trainable',
                                          shape=[self.batch_size, self.latent_dim],
                                          initializer=tf.truncated_normal_initializer,
                                          trainable=True)

            # Define placeholder for training steps in test procedure
            x_pl = tf.placeholder(tf.float32, shape=x_tensor.shape.as_list(), name='x_placeholder')
            y_pl = tf.placeholder(tf.float32, shape=y_tensor.shape.as_list(), name='y_placeholder')

        # Build computational graphs for encoding 
        x_hat = self.decoder(z_trainable, is_training=False)
        with tf.name_scope('anomaly_score'):
            # Residual score 
            with tf.name_scope('residual_error'):
                delta = tf.layers.flatten(x_hat - x_pl)
                residual_error = tf.norm(delta, ord=order_norm, axis=1, keepdims=False)
                mean_residual_error = tf.reduce_sum(residual_error)

            # Discriminative score 
            y_real, features_real = self.discriminator(x_pl, is_training=False)
            y_recon, features_recon = self.discriminator(x_hat, is_training=False)

            with tf.name_scope('discriminative_error'):
                delta_features = tf.layers.flatten(features_real - features_recon)
                disc_feature_error = tf.norm(delta_features, ord=order_norm, axis=1, keepdims=False)

            disc_score = -tf.squeeze(tf.log(y_real), axis=1) if scoring_method == 'xent' else disc_feature_error
            anomaly_score = (1-disc_weight)*residual_error + disc_weight*disc_score

        tensor_to_run = {'x_real' : x_pl,
                         'x_hat' : x_hat,
                         'label' : y_pl,
                         'score' : anomaly_score}

        with tf.variable_scope('test'):
            test_step = tf.get_variable(name='step',
                                             shape=[],
                                             dtype=tf.int64,
                                             initializer=tf.zeros_initializer,
                                             trainable=False)
            learning_rate = tf.train.piecewise_constant(test_step,
                                                        boundaries=[200, 300],
                                                        values=[0.01, 0.001, 0.0005])

            # Define optimizer to find latent embedding of given data points 
            optimizer_for_encoding = tf.train.AdamOptimizer(learning_rate)
            train_for_encoding = optimizer_for_encoding.minimize(mean_residual_error,
                                                                 global_step=test_step,
                                                                 var_list=[z_trainable],
                                                                 name='train_for_encoding')

        # Initilizers for variables for test
        init_op_test = tf.variables_initializer(tf.global_variables('test'))

        def anomaly_score_fn():
            """ compute anomaly scores and returns reconstructed images

            Returns:
                output_values: A dictionary which maps 'x_real' to real samples of 4d-array of size [batch_size, height, width, channel],
                'x_hat' to reconstructed samples of the same shape with real samples,
                'label' to binary labels (1d-array) indicating anomalousness: 1 is abnormal and 0 is normal,
                'score' to anomaly scores (1d-array): the degree of how much samples are abnormal
            """
            # Initailize variables for encoding
            sess = self.get_default_session()
            sess.run(init_op_test)

            # Define feed dictionary for evaluation
            x_feed, y_feed = sess.run([x_tensor, y_tensor])
            feed_dict = {x_pl:x_feed, y_pl:y_feed}

            batch_size_runtime = x_feed.shape[0]
            if self.batch_size != batch_size_runtime:
                dummy_size = self.batch_size - batch_size_runtime
                x_dummy = np.zeros([dummy_size, *x_feed.shape[1:]], dtype=np.float32)
                feed_dict[x_pl] = np.concatenate([x_feed, x_dummy], axis=0)

            # Update random latent points to minimize residual error
            for _ in range(self.iters_for_encoding):
                sess.run(train_for_encoding, feed_dict)

            output_values = sess.run(tensor_to_run, feed_dict)
            for k, v in output_values.items():
                output_values[k] = v[:batch_size_runtime]

            return output_values

        return anomaly_score_fn

