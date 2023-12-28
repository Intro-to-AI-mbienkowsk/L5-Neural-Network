from NeuralNetwork import NoAutogradNeuralNetwork
from constants import MNIST_NN_LAYERS
import numpy as np


class MnistNeuralNetwork(NoAutogradNeuralNetwork):
    def __init__(self, **kwargs):
        kwargs.pop("layer_sizes")
        super().__init__(layer_sizes=MNIST_NN_LAYERS, **kwargs)

    def classify(self, x):
        # todo: vectorize
        return np.array([np.argmax(self.output(image)) for image in x])
