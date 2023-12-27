from NeuralNetwork import NoAutogradNeuralNetwork
from constants import MNIST_NN_LAYERS
import numpy as np


class MnistNeuralNetwork(NoAutogradNeuralNetwork):
    def __init__(self, **kwargs):
        kwargs.pop("layer_sizes")
        super().__init__(layer_sizes=MNIST_NN_LAYERS, **kwargs)

    def classify(self, x):
        # todo: -1 label for the ones not classified?
        output_vector = self.output(x)
        # index of the largest value out of all outputs
        return np.where(output_vector == np.max(output_vector))[0]
