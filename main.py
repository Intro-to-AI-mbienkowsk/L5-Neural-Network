from MnistNN import MnistNeuralNetwork
from NeuralNetwork import NeuralNetwork
from constants import MNIST_NN_LAYERS
from enums import TrainingAlgorithm
from util import *


def main():
    (train_x, train_y), (test_x, test_y) = import_data()
    net = MnistNeuralNetwork(layer_sizes=MNIST_NN_LAYERS, activation_function=NeuralNetwork.sigmoid, epochs=10,
                             minimalization_algorithm=TrainingAlgorithm.BGD, learning_rate=.001)
    net.fit(train_x, train_y)
    classification = net.classify(test_x)
    display_confusion_matrix(classification, train_y)


if __name__ == "__main__":
    main()
