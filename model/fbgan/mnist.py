import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import conv2d, conv2d_transpose, fully_connected, \
        batch_norm, dropout

# Custom libs
from model.fbgan.base import default_arg_scope, BaseFBGAN

def leaky_relu(features):
    return tf.nn.leaky_relu(features, alpha=0.1)

class FBGAN(BaseFBGAN):
    """ FBGAN model for MNIST dataset. discriminator_forward, discriminator_backward, decoder and encoder function should be defined here """

    def __init__(self, *args, **kwargs):
        super(FBGAN, self).__init__(*args, **kwargs)

    def discriminator_forward(self,
                              x_tensor,
                              is_training):
        """ Build discriminator to compute probabilities that given samples x are from the true data distribution

        Args:
            x_tensor: Tensor of shape [batch, height, width, channel], a batch of data points
            is_training: Boolean, whether the output of this function will be used for training or inference

        Returns:
            probs: Tensor of shape [batch, 1], probability that given data points are from a true data distribution
            disc_features: Tensor of shape [batch, feature_dim], discriminative feature vectors
        """
        with arg_scope(default_arg_scope(is_training, activation_fn=leaky_relu)):
            with tf.variable_scope('discriminator_forward', reuse=tf.AUTO_REUSE):
                h = conv2d(x_tensor, num_outputs=64, kernel_size=4, stride=2, scope='conv1')
                h = conv2d(h, num_outputs=128, kernel_size=4, stride=2, scope='conv2')
                h = conv2d(h, num_outputs=32, kernel_size=4, stride=1, scope='conv3')
                disc_features = tf.layers.flatten(h, name='flatten')
                logits = fully_connected(disc_features, num_outputs=1, activation_fn=None, normalizer_fn=None, scope='logit')
                probs = tf.math.sigmoid(logits, name='sigmoid')
        return probs, disc_features

    def discriminator_backward(self,
                               z_tensor,
                               is_training):
        """ Build discriminator to compute probabilities that given samples z are from the latent prior distribution

        Args:
            z_tensor: Tensor of shape [batch, latent_dim], a batch of latent points
            is_training: Boolean, whether the output of this function will be used for training or inference

        Returns
            probs: Tensor of shape [batch, 1], probability that given data points are from a true data distribution
            disc_features: Tensor of shape [batch, feature_dim], discriminative feature vectors
        """
        with arg_scope(default_arg_scope(is_training, activation_fn=leaky_relu)):
            with tf.variable_scope('discriminator_backward', reuse=tf.AUTO_REUSE):
                    h = fully_connected(z_tensor, num_outputs=512, normalizer_fn=None, scope='fc1')
                    h = dropout(h, keep_prob=0.5, is_training=is_training, scope='dropout1')
                    disc_features = fully_connected(h, num_outputs=7*7*32, normalizer_fn=None, scope='fc2')
                    logits = fully_connected(disc_features, num_outputs=1, activation_fn=None, normalizer_fn=None, scope='logit')
                    probs = tf.math.sigmoid(logits, name='sigmoid')
        return probs, disc_features

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
        with arg_scope(default_arg_scope(is_training)):
            with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
                h = fully_connected(z_tensor, num_outputs=7*7*128, scope='transform')
                h = tf.reshape(h, shape=[-1, 7, 7, 128], name='reshape')
                h = conv2d_transpose(h, num_outputs=64, kernel_size=4, stride=2, scope='conv1_transpose')
                h = conv2d_transpose(h, num_outputs=1, kernel_size=4, stride=2,
                        activation_fn=None, normalizer_fn=None, scope='conv2_transpose')
                output_tensor = tf.math.tanh(h, name='tanh')
        return output_tensor

    def encoder(self,
                x_tensor,
                is_training):
        """ Encoder which maps a data point to a latent point

        Args:
            x_tensor: Tensor of shape [batch, height, width, channel], a batch of data points
            is_training: Boolean, whether the output of this function will be used for training or inference

        Returns:
            z_enc: Tensor of shape [batch, latent_dim]
            features: Tensor of shape [batch, feature_dim]
        """
        with arg_scope(default_arg_scope(is_training)):
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                h = conv2d(x_tensor, num_outputs=32, kernel_size=3, stride=1, scope='conv1')
                h = conv2d(h, num_outputs=64, kernel_size=3, stride=2, scope='conv2')
                h = conv2d(h, num_outputs=128, kernel_size=3, stride=2, scope='conv3')
                h = tf.layers.flatten(h, name='flatten')
                z_enc = fully_connected(h, num_outputs=self.latent_dim,
                        activation_fn=None, normalizer_fn=None, scope='encoding')
        return z_enc

