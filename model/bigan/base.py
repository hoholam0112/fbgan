"""
    This source code has been implemented following description in papers
    in which BiGAN has been given its birth:

    Adversarial Feature Learning, 2016, ICLR - https://arxiv.org/abs/1605.09782
    Adversarially Learned Inference, 2016, ICLR - https://arxiv.org/abs/1606.00704
"""
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

class BaseBiGAN:
    def __init__(self,
                 x_tensor_train,
                 z_tensor_train,
                 batch_size,
                 latent_dim,
                 learning_rate):
        """ Initialize BiGAN object for training and evaluation of BiGAN model

        Args:
            x_tensor_train: Tensor of shape [batch, *input_shape]. Batch of input data from true data distribution.
            z_tensor_train: Tensor of shape [batch, latent_dim]. Batch of latent samples from pre-defined prior distribution.
            batch_size: Integer, batch size used for training
            latent_dim: Integer, size of latent variable dimension
            learning_rate: Tensor or scala. Learning rate to be used for optimization.
        """
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        # Build a computational graph for training a model
        z_enc = self.encoder(x_tensor_train, is_training=True)
        y_enc, features_enc = self.discriminator(x_tensor_train, z_enc, is_training=True)
        x_fake = self.decoder(z_tensor_train, is_training=True)
        y_dec, features_fake = self.discriminator(x_fake, z_tensor_train, is_training=True)

        # Defined loss function
        self.loss = {}
        with tf.name_scope('loss_functions'):
            with tf.name_scope('discriminator_loss'):
                self.loss['discriminator'] = tf.negative(tf.reduce_mean(tf.log(y_enc + _TINY) + tf.log(1.0 - y_dec + _TINY)))
            with tf.name_scope('generator_loss'):
                self.loss['generator'] = tf.negative(tf.reduce_mean(tf.log(y_dec + _TINY)))
            with tf.name_scope('encoder_loss'):
                self.loss['encoder'] = tf.negative(tf.reduce_mean(tf.log(1.0 - y_enc + _TINY)))

        self.train_op = {}
        self.ema = tf.train.ExponentialMovingAverage(decay=0.999)
        with tf.name_scope('train_ops'):
            for scope in ['discriminator', 'generator', 'encoder']:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
                with tf.control_dependencies(update_ops):
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, name='Adam_{}'.format(scope))
                    self.train_op[scope] = optimizer.minimize(self.loss[scope], var_list=tf.trainable_variables(scope))

                # Create ops for exponential moving average 
                with tf.control_dependencies([self.train_op[scope]]):
                    self.train_op[scope] = tf.group(self.ema.apply(tf.trainable_variables(scope)))

        # Create variables for summary
        self.summary_vars = {}
        with tf.variable_scope('summary'):
            self.summary_vars['discriminator'] = TFScalarVariableWrapper(0, tf.float32, 'discriminator_loss')
            self.summary_vars['generator'] = TFScalarVariableWrapper(0, tf.float32, 'generator_loss')
            self.summary_vars['encoder'] = TFScalarVariableWrapper(0, tf.float32, 'encoder_loss')

        # Define variables to be restored from .ckpt files
        self.vars_to_restore = self.ema.variables_to_restore()

    def get_default_session(self):
        sess = tf.get_default_session()
        if sess is None:
            raise RuntimeError('this method must be called within tf.Session context manager')
        return sess

    def train_on_batch(self):
        """ Run training operations on a batch of training set """
        sess = self.get_default_session()
        loss_val = {}
        for scope in ['discriminator', 'generator', 'encoder']:
            loss_val[scope], _ = sess.run([self.loss[scope], self.train_op[scope]])
        return loss_val

    def discriminator(self,
                      x_tensor,
                      z_tensor,
                      is_training):
        """ Build discriminator to compute probabilities that given samples are real

        Args:
            x_tensor: Tensor of shape [batch, height, width, channel], a batch of data points
            z_tensor: Tensor of shape [batch, latent_dim], a batch of latent points
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

    def encoder(self,
                x_tensor,
                is_training):
        """ Encoder which maps a data point to a latent point

        Args:
            x_tensor: Tensor of shape [batch, height, width, channel], a batch of data points
            is_training: Boolean, whether the output of this function will be used for training or inference

        Returns:
            output_tensor: Tensor of shape [batch, latent_dim]
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

        # Build computational graphs to compute anomaly score and reconstructed images 
        z_enc = self.encoder(x_tensor, is_training=False)
        x_hat = self.decoder(z_enc, is_training=False)
        y_real, features_real = self.discriminator(x_tensor, z_enc, is_training=False)
        y_recon, features_recon = self.discriminator(x_hat, z_enc, is_training=False)

        with tf.name_scope('anomaly_score'):
            # Residual score 
            with tf.name_scope('residual_error'):
                delta = tf.layers.flatten(x_hat - x_tensor)
                residual_error = tf.norm(delta, ord=order_norm, axis=1, keepdims=False)

            with tf.name_scope('discriminative_error'):
                delta_features = tf.layers.flatten(features_real - features_recon)
                disc_feature_error = tf.norm(delta_features, ord=order_norm, axis=1, keepdims=False)

            if scoring_method == 'xent':
                disc_score = tf.squeeze(tf.negative(tf.log(y_real + _TINY)), axis=1)
            else:
                disc_score = disc_feature_error
            anomaly_score = (1.0 - disc_weight)*residual_error + disc_weight*disc_score

        tensor_to_run = {'x_real' : x_tensor,
                         'x_hat' : x_hat,
                         'label' : y_tensor,
                         'score' : anomaly_score}

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
            output_values = sess.run(tensor_to_run)
            return output_values

        return anomaly_score_fn


