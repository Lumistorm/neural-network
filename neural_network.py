import numpy as np
np.random.seed(3)


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, weighted_sum):
        self.output = 1 / (1 + np.exp(-weighted_sum))
        return self.output

    def backward(self, gradiant):
        return gradiant * self.output * (1 - self.output)


class ReLU:
    def __init__(self):
        self.output = None

    def forward(self, weighted_sum):
        self.output = np.maximum(weighted_sum, 0)
        return self.output

    def backward(self, gradiant):
        return gradiant * (self.output > 0)


class SoftMax:
    def __init__(self):
        self.output = None

    def forward(self, weighted_sum):
        exp_shift = np.exp(weighted_sum - np.max(weighted_sum))
        self.output = exp_shift / np.sum(exp_shift, axis=0, keepdims=True)
        return self.output

    def backward(self, gradient):
        return gradient


class Linear:
    def __init__(self, in_features, out_features):
        self.weights = np.random.uniform(-0.5, 0.5, (out_features, in_features))
        self.bias = np.zeros((out_features, 1))
        self.inputs = None
        self.delta_weights = None
        self.delta_bias = None

    def forward(self, inputs):
        self.inputs = inputs

        return self.weights @ inputs + self.bias

    def backward(self, gradient):
        self.delta_weights = gradient @ self.inputs.T
        self.delta_bias = np.sum(gradient, axis=1, keepdims=True)

        return self.weights.T @ gradient

    def update_parameters(self, learning_rate):
        self.weights -= learning_rate * self.delta_weights
        self.bias -= learning_rate * self.delta_bias


class NeuralNetwork:
    def __init__(self):
        self.layers = [
            Linear(784, 16),
            ReLU(),
            Linear(16, 16),
            ReLU(),
            Linear(16, 10),
            SoftMax(),
        ]

    def train(self, inputs, targets, epochs, batch_size, learning_rate):
        num_inputs = len(inputs)
        for epoch in range(epochs):
            correct = 0

            indices = np.random.permutation(num_inputs)
            inputs_shuffled = inputs[indices]
            targets_shuffled = targets[indices]

            for batch_start in range(0, num_inputs, batch_size):
                batch_end = batch_start + batch_size

                input_batch = inputs_shuffled[batch_start:batch_end].T
                target_batch = targets_shuffled[batch_start:batch_end].T
                prediction = self.forward_prop(input_batch)

                # add correct predictions
                predicted_result = np.argmax(prediction, axis=0)
                target_result = np.argmax(target_batch, axis=0)
                correct += np.sum(predicted_result == target_result)

                # calculate error value
                mean_square_error = prediction - target_batch

                self.backward_prop(mean_square_error / batch_size)
                self.update_layers(learning_rate)

            print(round(correct / len(inputs), 4))

    def forward_prop(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    def backward_prop(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def update_layers(self, learning_rate):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.update_parameters(learning_rate)

    def save(self, path):
        model_data = {}

        for index, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                model_data[f'w_{index}'] = layer.weights
                model_data[f'b_{index}'] = layer.bias

        np.savez(path, **model_data)

    def load(self, path):
        with np.load(path) as f:
            model_data = {key: f[key] for key in f.files}

        for index, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                layer.weights = model_data[f'w_{index}']
                layer.bias = model_data[f'b_{index}']
