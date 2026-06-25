import numpy as np

from ..layers import Layer, LAYER_REGISTRY
from ..losses.losses import cross_entropy_loss


class Sequential:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def train(
            self, inputs: np.ndarray, targets: np.ndarray,
            *, batch_size: int, epochs: int, learning_rate: float,
    ) -> None:

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
                total_loss += cross_entropy_loss(predictions, target_batch) * current_batch_size

                # calculate loss gradient
                loss_gradient = (predictions - target_batch) / current_batch_size

                # backward_propagation
                self.backward_prop(loss_gradient)
                self.update_layers(learning_rate)

            total_loss /= num_inputs
            print(f'Total Loss: {total_loss:.4f}')
            print(f'Accuracy: {(correct / len(inputs)):.4f}')

    def forward_prop(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs

    def backward_prop(self, gradient: np.ndarray) -> None:
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def update_layers(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.update_parameters(learning_rate)

    def save(self, path: str) -> None:
        model_dict = {}

        for index, layer in enumerate(self.layers):
            model_dict[f'layer_{index}'] = {
                'type': type(layer).__name__.lower(),
                'config': layer.get_config(),
                'parameters': layer.parameters(),
            }

        np.savez(path, **model_dict, allow_pickle=True)

    @classmethod
    def load(cls, path: str) -> Sequential:
        with np.load(path, allow_pickle=True) as f:
            model_dict = {key: f[key].item() for key in f.files}

        layers = []

        for layer in model_dict.values():
            layer_type = layer['type']
            config = layer['config']
            parameters = layer['parameters']

            layer_instance = LAYER_REGISTRY[layer_type].from_config(config)
            layer_instance.set_parameters(parameters)

            layers.append(layer_instance)

        return cls(layers)
