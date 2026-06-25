import numpy as np
from layer import LAYER_REGISTRY
from linear import Linear
from activations import ReLU, SoftMax


class NeuralNetwork:
    def __init__(self):
        self.layers = [
            Linear(784, 512),
            ReLU(),
            Linear(512, 256),
            ReLU(),
            Linear(256, 10),
            SoftMax(),
        ]

    def train(self, inputs, targets, epochs, batch_size, learning_rate):
        num_inputs = len(inputs)

        for epoch in range(epochs):
            correct = 0
            total_loss = 0

            indices = np.random.permutation(num_inputs)
            inputs_shuffled = inputs[indices]
            targets_shuffled = targets[indices]

            for batch_start in range(0, num_inputs, batch_size):
                # get batch data
                input_batch = inputs_shuffled[batch_start:batch_start + batch_size]
                target_batch = targets_shuffled[batch_start:batch_start + batch_size]
                current_batch_size = input_batch.shape[0]

                # froward propagation
                predictions = self.forward_prop(input_batch)

                # add correct predictions
                predicted_classes = np.argmax(predictions, axis=1)
                target_result = np.argmax(target_batch, axis=1)
                correct += np.sum(predicted_classes == target_result)

                # calculate loss
                total_loss += self.cross_entropy_loss(predictions, target_batch)

                # calculate loss gradient
                loss_gradient = (predictions - target_batch) / current_batch_size

                # backward_propagation
                self.backward_prop(loss_gradient)
                self.update_layers(learning_rate)

            total_loss /= num_inputs
            print(f'Total Loss: {total_loss:.4f}')
            print(f'Accuracy: {(correct / len(inputs)):.4f}')

    def cross_entropy_loss(self, prediction, target):
        epsilon = 1e-15
        clipped_prediction = np.clip(prediction, epsilon, 1.0 - epsilon)
        loss = -np.sum(target * np.log(clipped_prediction))

        return np.mean(loss)

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
        model_dict = {}

        for index, layer in enumerate(self.layers):
            model_dict[f'layer_{index}'] = {
                'type': type(layer).__name__,
                'config': layer.get_config(),
                'parameters': layer.parameters(),
            }

        np.savez(path, **model_dict, allow_pickle=True)

    def load(self, path):
        with np.load(path, allow_pickle=True) as f:
            model_dict = {key: f[key].item() for key in f.files}

        self.layers = []

        for layer in model_dict.values():
            layer_type = layer['type']
            config = layer['config']
            parameters = layer['parameters']

            layer_instance = LAYER_REGISTRY[layer_type].from_config(config)
            if parameters:
                layer_instance.load_parameters(parameters)

            self.layers.append(layer_instance)
