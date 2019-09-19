"""
    This source code has been implemented following description in papers
    in which BiGAN has been given its birth:

    Adversarial Feature Learning, 2016, ICLR - https://arxiv.org/abs/1605.09782
    Adversarially Learned Inference, 2016, ICLR - https://arxiv.org/abs/1606.00704
"""

# Import required frameworks 
import tensorflow as tf

# Import custom modules 


class BiGANModel:
    """ Model class for Bidirectional Generative Adversarial Networks """

    def __init__(self,
                 generator,
                 encoder,
                 discriminator):
        """ Initialize a BiGANModel object

        Args:
            generator: A callable Model object whose argument is 2d tensor of shape [batch_size, latent_dim] and is_training
            encoder: A callable Model object whose argument is 2d tensor of shape [batch_size, *input_shape] and is_training
            discriminator: A callable Model object whose arguments is a tensor of shape [batch_size, *input_shape] and is_training

        Returns:
            A BiGANModel object
        """

        self.generator = generator
        self.encoder = encoder
        self.discriminator = discriminator

    def loss(self):
        """ Compute Loss fuction to train a BiGAN model """
        self.generator(is_training=True)



