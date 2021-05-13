import numpy as np

# sigmoid activation function
def sigmoid(x):
    # f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def derivative_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


# mean squared error loss function for our network
# goal is to minimize loss of our network
def mse_loss(y_true, y_pred):
    # inputs are np arrays of equal length
    return ((y_true - y_pred) ** 2).mean()


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        result = np.dot(self.weights, inputs) + self.bias
        return sigmoid(result)


class NeuralNetwork:
    '''
    Class for a neural network with:
        - 2 inputs
        - one hidden layer with 2 neurons (h1, h2)
        - one output layer with 1 neuron (o1)
    Each neuron has weights w = [0, 1], and bias b = 0
    '''
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, inputs):
        h1_out = self.h1.feedforward(inputs)
        h2_out = self.h2.feedforward(inputs)

        out_o1_input = np.array([h1_out, h2_out])
        o1_out = self.o1.feedforward(out_o1_input)
        
        return o1_out
    