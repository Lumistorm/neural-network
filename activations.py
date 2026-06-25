import numpy as np


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, output_gradient):
        return output_gradient * self.output * (1 - self.output)


class ReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(inputs, 0)
        return self.output

    def backward(self, output_gradient):
        return output_gradient * (self.output > 0)


class SoftMax:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exp_shift = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_shift / np.sum(exp_shift, axis=1, keepdims=True)
        return self.output

    def backward(self, output_gradient):
        return output_gradient
