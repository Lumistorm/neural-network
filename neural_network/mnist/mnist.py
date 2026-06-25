import numpy as np


def preprocess_mnist_train():
    with np.load('mnist.npz') as f:
        x_train = f['x_train'].astype(np.float32) / 255
        y_train = f['y_train']

    x_train.shape = (x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    y_train = np.eye(10)[y_train]

    return x_train, y_train


def preprocess_mnist_test():
    with np.load('mnist.npz') as f:
        x_test = f['x_test'].astype(np.float32) / 255
        y_test = f['y_test']

    x_test.shape = (x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    y_test = np.eye(10)[y_test]

    return x_test, y_test
