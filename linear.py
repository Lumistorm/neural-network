import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        limit = np.sqrt(6 / in_features)
        self.weights = np.random.uniform(-limit, limit, (in_features, out_features))
        self.biases = np.zeros((out_features,))

        self.inputs = None
        self.weight_gradients = None
        self.bias_gradients = None

    def forward(self, inputs):
        self.inputs = inputs

        return inputs @ self.weights + self.biases

    def backward(self, output_gradient):
        self.weight_gradients = self.inputs.T @ output_gradient
        self.bias_gradients = np.sum(output_gradient, axis=0)

        return output_gradient @ self.weights.T

    def update_parameters(self, learning_rate):
        self.weights -= learning_rate * self.weight_gradients
        self.biases -= learning_rate * self.bias_gradients