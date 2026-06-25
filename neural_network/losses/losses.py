import numpy as np


EPSILON = 1e-15


def cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    clipped_prediction = np.clip(predictions, EPSILON, 1.0 - EPSILON)
    loss = -np.sum(targets * np.log(clipped_prediction), axis=1)

    return np.sum(loss)
