import numpy as np

from ..layers.layer import Layer, register_layer


@register_layer
class Sigmoid(Layer):
    def __init__(self) -> None:
        super().__init__()

        self.output = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = 1 / (1 + np.exp(-inputs))

        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * self.output * (1 - self.output)