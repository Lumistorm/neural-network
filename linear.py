import numpy as np
from layer import Layer, register_layer


@register_layer
class Linear(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

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

    def parameters(self):
        return {
            'weights': self.weights,
            'biases': self. biases
        }

    def load_parameters(self, parameters):
        self.weights = parameters['weights']
        self.biases = parameters['biases']

    def get_config(self):
        return {
            'in_features': self.in_features,
            'out_features': self.out_features
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            in_features=config['in_features'],
            out_features=config['out_features']
        )
