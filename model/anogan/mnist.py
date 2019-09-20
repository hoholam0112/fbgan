import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import conv2d, conv2d_transpose, fully_connected, batch_norm

# Custom libs
from anogan.base import default_arg_scope, BaseAnoGAN

class AnoGAN(BaseAnoGAN):
    """ AnoGAN model for MNIST dataset. discriminator and decoder function should be defined here """

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)

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
        with arg_scope(default_arg_scope(is_training)):
            with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
                h = conv2d(tensor_x, num_outputs=64, kernel_size=4, stride=2, scope='conv1')
                h = conv2d(h, num_outputs=128, kernel_size=4, stride=2, scope='conv2')
                h = conv2d(h, num_outputs=32, kernel_size=4, stride=1, scope='conv3')
                disc_features = tf.layers.flatten(h, name='flatten')
                h = fully_connected(disc_features, num_outputs=1, activation_fn=None, normalizer_fn=None, 'output')
                probs = tf.math.sigmoid(h, name='sigmoid')
        return probs, disc_features

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
        with arg_scope(default_arg_scope(is_training)):
            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                h = fully_connected(tensor_z, num_outputs=7*7*128, scope='linear_transform')
                h = conv2d_transpose(h, num_outputs=64, kernel_size=4, stride=2, scope='trans_conv1')
                h = conv2d_transpose(h, num_outputs=1, kernel_size=4, stride=2, scope='trans_conv2')
                output_tensor = tf.math.tanh(h, name='tanh')
        return output_tensor

