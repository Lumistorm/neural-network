import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.weights1 = np.random.uniform(-0.5, 0.5, (16, 784))
        self.biases1 = np.random.uniform(-0.5, 0.5, (16, 1))

        self.weights2 = np.random.uniform(-0.5, 0.5, (16, 16))
        self.biases2 = np.random.uniform(-0.5, 0.5, (16, 1))

        self.weights3 = np.random.uniform(-0.5, 0.5, (10, 16))
        self.biases3 = np.random.uniform(-0.5, 0.5, (10, 1))

    def train(self, inputs, targets, epochs, batch_size, learning_rate):
        num_inputs = len(inputs)
        for epoch in range(epochs):
            indices = np.random.permutation(num_inputs)
            inputs_shuffled = inputs[indices]
            targets_shuffled = targets[indices]

            for batch_start in range(0, batch_size, batch_size):
                bath_end = batch_start + batch_size

                input_batch = inputs_shuffled[batch_start:bath_end]
                target_batch = targets_shuffled[batch_start:bath_end]
                outputs = self.forward(input_batch)

                loss = self.loss(outputs, target_batch)

    def forward(self, inputs):
        weighted_sums1 = self.weights1 @ inputs + self.biases1
        self.hidden_outputs1 = 1 / (1 + np.exp(-weighted_sums1))

        weighted_sums2 = self.weights2 @ self.hidden_outputs1 + self.biases2
        self.hidden_outputs2 = 1 / (1 + np.exp(-weighted_sums2))

        outputs = self.weights3 @ self.hidden_outputs2 + self.biases3
        outputs = 1 / (1 + np.exp(-outputs))

        return outputs

    def backpropagation(self, outputs, targets):
        delta_output = (outputs - targets) * (outputs * (1 - outputs))
        grad_weights3 = self.hidden_outputs2.T @ delta_output

        delta_hidden2 = (grad_weights3.T @ delta_output) * (self.hidden_outputs2 * (1 - self.hidden_outputs2))
        grad_weights2 = self.hidden_output1.T @ delta_output

        delta_hidden1 = (grad_weights2.T @ delta_output) * (self.hidden_outputs1 * (1 - self.hidden_outputs1))
        grad_weights2 = self.hidden_output1.T @ delta_output

        self.weights1 -=
        self.biases1 = np.random.uniform(-0.5, 0.5, (16, 1))

        self.weights2 = np.random.uniform(-0.5, 0.5, (16, 16))
        self.biases2 = np.random.uniform(-0.5, 0.5, (16, 1))

        self.weights3 = np.random.uniform(-0.5, 0.5, (10, 16))
        self.biases3 = np.random.uniform(-0.5, 0.5, (10, 1))


