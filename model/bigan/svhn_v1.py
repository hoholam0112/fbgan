import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import conv2d, conv2d_transpose, fully_connected, \
        batch_norm, dropout

# Custom libs
from model.bigan.base import default_arg_scope, BaseBiGAN

def leaky_relu(features):
    return tf.nn.leaky_relu(features, alpha=0.1)

class BiGAN(BaseBiGAN):
    """ BiGAN model for MNIST dataset. discriminator, decoder and encoder function should be defined here """

    def __init__(self, *args, **kwargs):
        super(BiGAN, self).__init__(*args, **kwargs)

    def discriminator(self,
                      x_tensor,
                      z_tensor,
                      is_training):
        """ Build discriminator to compute probabilities that given samples (x, z) are from the encoder

        Args:
            x_tensor: Tensor of shape [batch, height, width, channel], a batch of data points
            z_tensor: Tensor of shape [batch, latent_dim], a batch of latent points
            is_training: Boolean, whether the output of this function will be used for training or inference

        Returns:
            probs: Tensor of shape [batch, 1], probability that given data points are from a true data distribution
            disc_features: Tensor of shape [batch, feature_dim], discriminative feature vectors
        """
        with arg_scope(default_arg_scope(is_training, activation_fn=leaky_relu)):
            with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
                with tf.variable_scope('network_x'):
                    h = conv2d(x_tensor, num_outputs=64, kernel_size=3, stride=2, scope='conv1')
                    h = conv2d(h, num_outputs=128, kernel_size=3, stride=2, scope='conv2')
                    h = conv2d(h, num_outputs=256, kernel_size=3, stride=2, scope='conv3')
                    h = conv2d(h, num_outputs=64, kernel_size=3, stride=1, scope='conv4')
                    x_features = tf.layers.flatten(h, name='flatten')
                with tf.variable_scope('network_z'):
                    z_features = fully_connected(z_tensor, num_outputs=256, normalizer_fn=None, scope='fc1')
                    z_features = fully_connected(z_tensor, num_outputs=512, normalizer_fn=None, scope='fc2')
                with tf.variable_scope('network_relation'):
                    total_features = tf.concat([x_features, z_features], axis=-1)
                    h = dropout(total_features, keep_prob=0.5, is_training=is_training, scope='dropout1')
                    h = fully_connected(h, num_outputs=1024, normalizer_fn=None, scope='fc1')
                    h = dropout(h, keep_prob=0.5, is_training=is_training, scope='dropout2')
                    disc_features = fully_connected(h, num_outputs=1024, normalizer_fn=None, scope='fc2')
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
                h = fully_connected(z_tensor, num_outputs=4*4*256, scope='transform')
                h = tf.reshape(h, shape=[-1, 4, 4, 256], name='reshape')
                h = conv2d_transpose(h, num_outputs=128, kernel_size=3, stride=2, scope='conv1_transpose')
                h = conv2d_transpose(h, num_outputs=64, kernel_size=3, stride=2, scope='conv2_transpose')
                h = conv2d_transpose(h, num_outputs=32, kernel_size=3, stride=2, scope='conv3_transpose')
                h = conv2d_transpose(h, num_outputs=3, kernel_size=3, stride=1,
                        activation_fn=None, normalizer_fn=None, scope='conv4_transpose')
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
        """
        with arg_scope(default_arg_scope(is_training)):
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                h = conv2d(x_tensor, num_outputs=64, kernel_size=3, stride=2, scope='conv1')
                h = conv2d(h, num_outputs=128, kernel_size=3, stride=2, scope='conv2')
                h = conv2d(h, num_outputs=256, kernel_size=3, stride=2, scope='conv3')
                h = conv2d(h, num_outputs=64, kernel_size=3, stride=1, scope='conv4')
                conv_features = tf.layers.flatten(h, name='flatten')
                z_enc = fully_connected(conv_features, num_outputs=self.latent_dim,
                        activation_fn=None, scope='encoding')
        return z_enc

