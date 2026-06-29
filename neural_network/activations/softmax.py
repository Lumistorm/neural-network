import numpy as np

from ..layers import Layer, register_layer


@register_layer
class Softmax(Layer):
    def __init__(self) -> None:
        super().__init__()

        self.output = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        exp_shift = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_shift / np.sum(exp_shift, axis=1, keepdims=True)

        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient
