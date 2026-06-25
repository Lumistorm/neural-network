import numpy as np
from layer import Layer, register_layer


class Activation(Layer):
    def __init__(self):
        super().__init__()


@register_layer
class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

        self.output = None

    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, output_gradient):
        return output_gradient * self.output * (1 - self.output)


@register_layer
class ReLU(Activation):
    def __init__(self):
        super().__init__()

        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(inputs, 0)
        return self.output

    def backward(self, output_gradient):
        return output_gradient * (self.output > 0)


@register_layer
class SoftMax(Activation):
    def __init__(self):
        super().__init__()

        self.output = None

    def forward(self, inputs):
        exp_shift = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_shift / np.sum(exp_shift, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient):
        return output_gradient
