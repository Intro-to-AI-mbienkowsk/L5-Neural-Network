import random
import numpy as np
from abc import ABC, abstractmethod
from src.constants import *


class NeuralNetwork(ABC):
    def __init__(self, layer_sizes: tuple[int], epochs: int, learning_rate: float):
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(y) for y in layer_sizes[1:]]
        self.epochs = epochs
        self.activation_function = NeuralNetwork.sigmoid
        self.learning_rate = learning_rate

    def output(self, x):
        y = x
        for b, w in zip(self.biases, self.weights):
            y = self.activation_function(np.dot(w, y) + b)
        return y

    @abstractmethod
    def calculate_gradient(self, train_x, train_y):
        ...

    @abstractmethod
    def fit(self, train_x, train_y):
        ...

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        return NeuralNetwork.sigmoid(z) * (1 - NeuralNetwork.sigmoid(z))


class NoAutogradNeuralNetwork(NeuralNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activation_derivative = NeuralNetwork.sigmoid_derivative

    def calculate_gradient(self, train_x, train_y):
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        weighted_inputs = []  # wx + b for each neuron
        activation = train_x
        activations = [activation]

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            weighted_inputs.append(z)
            activation = self.activation_function(z)
            activations.append(activation)

        delta = (activations[-1] - train_y) * self.activation_derivative(weighted_inputs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.outer(delta, activations[-2])

        for layer_idx in range(2, self.num_layers):
            # iterate back from the 2nd to last layer
            z = weighted_inputs[-layer_idx]
            delta = np.dot(self.weights[-layer_idx + 1].T, delta) * self.activation_derivative(z)
            grad_b[-layer_idx] = delta
            grad_w[-layer_idx] = np.outer(delta, activations[-layer_idx - 1])
        return grad_w, grad_b

    def fit(self, train_x, train_y):
        for i in range(self.epochs):
            print(f"Epoch {i + 1}")
            training_data = list(zip(train_x, train_y))
            random.shuffle(training_data)
            batches = [training_data[i:i + BGD_BATCH_SIZE] for i in range(0, len(training_data), BGD_BATCH_SIZE)]
            for batch in batches:
                self.train_batch(batch)

    def train_batch(self, batch):
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in batch:
            y_one_hot = np.eye(10)[y]
            partial_grad_w, partial_grad_b = self.calculate_gradient(x, y_one_hot)
            grad_w = [gw + pgw for gw, pgw in zip(grad_w, partial_grad_w)]
            grad_b = [gb + pgb for gb, pgb in zip(grad_b, partial_grad_b)]

        grad_w = [pgw * self.learning_rate / BGD_BATCH_SIZE for pgw in grad_w]
        grad_b = [pgb * self.learning_rate / BGD_BATCH_SIZE for pgb in grad_b]
        self.weights = [w - gw for w, gw in zip(self.weights, grad_w)]
        self.biases = [b - gb for b, gb in zip(self.biases, grad_b)]
