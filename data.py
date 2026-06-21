import numpy as np
import struct


def load_mnist():
    with (np.load('mnist.npz') as f):
        x_train = f['x_train'].shape + (1,)
        y_train = f['y_train'].shape + (1,)
        x_test = f['x_test'].shape + (1,)
        y_test = f['y_test'].shape + (1,)

    return (x_train, y_train), (x_test, y_test)


