import numpy as np
from typing import Type, Self


class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update_parameters(self, learning_rate: float) -> None:
        pass

    def parameters(self) -> dict:
        return {}

    def set_parameters(self, parameters: dict) -> None:
        pass

    def get_config(self) -> dict:
        return {}

    @classmethod
    def from_config(cls, config: dict) -> Self:
        return cls()


LAYER_REGISTRY = {}


def register_layer(cls):
    LAYER_REGISTRY[cls.__name__.lower()] = cls
    return cls
