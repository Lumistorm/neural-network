
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError

    def update_parameters(self, learning_rate):
        pass

    def parameters(self):
        return {}

    def load_parameters(self, parameters):
        pass

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls()


def register_layer(cls):
    LAYER_REGISTRY[cls.__name__] = cls
    return cls


LAYER_REGISTRY = {}



