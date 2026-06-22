import math
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
        self.weights_gradient = None
        self.bias_gradient = None

    def forward(self, inputs):
        self.inputs = inputs

        return self.weights @ inputs + self.bias

    def backward(self, gradient):
        self.weights_gradient = gradient @ self.inputs.T
        self.bias_gradient = np.sum(gradient, axis=1, keepdims=True)

        return self.weights.T @ gradient

    def update_parameters(self, learning_rate):
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient


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
        """
        Gradient descent through multiple epochs

        :param inputs:
        :param targets:
        :param epochs:
        :param batch_size:
        :param learning_rate:
        :return:
        """

        num_inputs = len(inputs)
        num_batches = math.ceil(num_inputs / batch_size)
        for epoch in range(epochs):
            correct = 0
            total_loss = 0

            indices = np.random.permutation(num_inputs)
            inputs_shuffled = inputs[indices]
            targets_shuffled = targets[indices]

            for batch_start in range(0, num_inputs, batch_size):
                batch_end = batch_start + batch_size

                # get batch data
                input_batch = inputs_shuffled[batch_start:batch_end].T
                target_batch = targets_shuffled[batch_start:batch_end].T
                current_batch_size = input_batch.shape[1]

                # froward propagation
                prediction = self.forward_prop(input_batch)

                # add correct predictions
                predicted_result = np.argmax(prediction, axis=0)
                target_result = np.argmax(target_batch, axis=0)
                correct += np.sum(predicted_result == target_result)

                # calculate loss
                epsilon = 1e-15
                clipped_prediction = np.clip(prediction, epsilon, 1.0 - epsilon)
                total_loss -= np.sum(target_batch * np.log(clipped_prediction)) / current_batch_size

                # calculate loss gradient
                loss_gradient = (prediction - target_batch) / current_batch_size

                # backward_propagation
                self.backward_prop(loss_gradient)
                self.update_layers(learning_rate)

            total_loss /= num_batches
            print(f'Total Loss: {total_loss:.4f}')
            print(f'Accuracy: {(correct / len(inputs)):.4f}')

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
