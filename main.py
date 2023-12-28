from src.NeuralNetwork import NoAutogradNeuralNetwork
from src.util import *
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description="Initialize the network with parameters")

    parser.add_argument('-H', '--HiddenLayerSizes', nargs='+', type=int,
                        help='The size of the hidden layers in the network', default=[30])
    parser.add_argument('-E', '--Epochs', type=int, help='The number of epochs for the model to train over', default=10)
    parser.add_argument('-L', '--LearningRate', type=float, help='The learning rate of the model', default=2)
    parser.add_argument('-T', '--NumTrainingImages', type=int, help='The number of training images for the model',
                        default=15000)

    args = parser.parse_args()

    (train_x, train_y), (test_x, test_y) = import_data()
    net = NoAutogradNeuralNetwork(layer_sizes=[784] + args.HiddenLayerSizes + [10],
                                  epochs=args.Epochs, learning_rate=args.LearningRate)
    net.fit(train_x, train_y)
    classification = net.classify(test_x)
    display_confusion_matrix(classification, test_y)


if __name__ == "__main__":
    main()
