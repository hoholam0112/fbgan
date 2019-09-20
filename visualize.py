import matplotlib.pyplot as plt
import numpy as np

def grid_plot(train_config, images, labels, num_rows=5, num_cols=5, show=False):
    """ Plotting grid plot to show images

    Args:
        train_config: TrainConfig object which contains training configurations such as hyperparameters
        images: 4d numpy array of shape [batch, height, width, channel]
        labels: 1d numpy array. anomaly labels.
        show: Boolean, whether to call plt.show()
    """
    plt.figure()
    for i in range(num_rows*num_cols):
        plt.subplot(num_rows, num_cols, i+1)
        if train_config.dataset == 'mnist':
            plt.imshow(np.squeeze(images[i]), cmap='gray')
        else:
            plt.imshow(images[i])
        plt.title(labels[i])
        plt.axis('off')

    if show:
        plt.show()
