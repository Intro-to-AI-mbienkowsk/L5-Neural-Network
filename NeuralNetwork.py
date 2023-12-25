import random
import numpy as np
from abc import ABC, abstractmethod
from enums import *
from autograd import grad


class NeuralNetwork(ABC):
    def __init__(self, layer_sizes: tuple[int], activation_function: callable, epochs: int,
                 minimalization_algorithm: LearningAlgorithm, learning_rate: float):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.activation_function = activation_function
        self.epochs = epochs
        self.learning_rate = learning_rate

    def output(self, x):
        y = x
        for b, w in self.biases, self.weights:
            y = self.activation_function(np.dot(y, w) + b)
        return y

    @abstractmethod
    def calculate_gradient(self, train_x, train_y):
        ...

    @abstractmethod
    def fit(self, train_x, train_y):
        ...

    @abstractmethod
    def test(self, test_x, test_y):
        ...

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))


class NoAutogradNeuralNetwork(NeuralNetwork):
    def __init__(self, **kwargs):
        # todo: is this good practice?
        super().__init__(**kwargs)
        self.activation_derivative = grad(self.activation_function)

    def calculate_gradient(self, train_x, train_y):
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        weighted_inputs = []  # wx + b for each neuron
        activation = train_x
        activations = [activation]  # act_function(wx+b)

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            weighted_inputs.append(z)
            activations.append(self.activation_function(z))

        delta = (self.cost_derivative(activations[-1], train_y) *
                 self.activation_derivative(weighted_inputs[-1]))

        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].T)

        for layer_idx in range(2, self.num_layers):
            # iterate back from the 2nd to last layer
            l = -layer_idx
            z = weighted_inputs[l]
            delta = np.dot(self.weights[l + 1].T, delta) * self.activation_derivative(z)
            grad_b[l] = delta
            grad_w[l] = np.dot(delta, activations[l + 1].T)
        return grad_w, grad_b
