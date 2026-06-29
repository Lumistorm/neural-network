import numpy as np

from ..layers import Layer, register_layer


@register_layer
class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()

        self.output = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = np.maximum(inputs, 0)

        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * (self.output > 0)
