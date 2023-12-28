from MnistNN import MnistNeuralNetwork
from NeuralNetwork import NeuralNetwork
from constants import *
from util import *


def main():
    (train_x, train_y), (test_x, test_y) = import_data()
    net = MnistNeuralNetwork(layer_sizes=MNIST_NN_LAYERS, activation_function=NeuralNetwork.sigmoid, epochs=EPOCHS,
                             learning_rate=LEARNING_RATE)
    net.fit(train_x, train_y)
    classification = net.classify(test_x)
    display_confusion_matrix(classification, test_y)


if __name__ == "__main__":
    main()
