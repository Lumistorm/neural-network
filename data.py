import numpy as np
import struct


def load_mnist():
    with np.load('mnist.npz') as f:
        x_train = f['x_train']
        y_train = f['y_train']

    x_train = x_train.astype(np.float32) / 255
    x_train.shape = (x_train.shape[0], x_train.shape[1] * x_train.shape[2])

    y_train = np.eye(10)[y_train]

    return x_train, y_train


