
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, inputs):
        pass

    def backward(self, output_gradient):
        pass

    def update_parameters(self, learning_rate):
        pass
