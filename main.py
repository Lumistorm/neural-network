from neural_network import NeuralNetwork
from data import load_mnist


if __name__ == '__main__':
    model = NeuralNetwork()
    x_train, y_train = load_mnist()

    model.train(x_train, y_train, 25, 50, 0.05)
    model.save('model_0')
