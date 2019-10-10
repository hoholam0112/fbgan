
from utils import TrainConfig

def complete_train_config(train_config):
    """ Complete train_config by replace None arguments with default parameters """
    if train_config.dataset == 'mnist':
        default_arguments = dict(latent_dim=35,
                                 batch_size=100,
                                 learning_rate=1e-3,
                                 end_epoch=100,
                                 iters_per_epoch=300)
    elif train_config.dataset == 'kdd':
        default_arguments = dict(latent_dim=32,
                                 batch_size=50,
                                 learning_rate=1e-5,
                                 end_epoch=100,
                                 iters_per_epoch=5000)
    elif train_config.dataset == 'svhn':
        default_arguments = dict(latent_dim=100,
                                 batch_size=100,
                                 learning_rate=1e-4,
                                 end_epoch=100,
                                 iters_per_epoch=400)

    for k, v in default_arguments.items():
        if getattr(train_config, k) is None:
            setattr(train_config, k, v)

