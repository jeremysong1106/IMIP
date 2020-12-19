import numpy as np
import matplotlib.pyplot as plt

from random import shuffle
from sklearn.datasets import load_iris
from LayersExercise.fullyConnected import FullyConnected
from LayersExercise.softMax import SoftMax
from LayersExercise.reLU import ReLU
# from LayersExercise import *


def main():  # test_iris_data, TestNeuralNetwork

    # load sample data
    iris_data = IrisData()
    categories = iris_data.categories
    input_size = iris_data.feature_dim

    # set network options
    learning_rate = 1e-4  # set same value for all layers here
    batch_size = iris_data.full_batch  # use all samples for this exercise

    # ---------------------------
    # Construction of the network
    # ---------------------------
    net = NeuralNetwork()

    net.data_layer = InputLayer(iris_data)

    # fully connected layer 1

    fcl_1 = FullyConnected(input_size, categories, batch_size, learning_rate)
    net.layers.append(fcl_1)
    net.layers.append(ReLU(categories, batch_size))

    # fully connected layer 2
    fcl_2 = FullyConnected(categories, categories, batch_size, learning_rate)
    net.layers.append(fcl_2)
    net.layers.append(ReLU(categories, batch_size))

    net.loss_layer = SoftMax(categories, batch_size)

    # -----------------------
    # Training of the network
    # -----------------------
    net.train(2000)

    # ---------------------------
    # Testing/running the network
    # ---------------------------
    # compute results for test data
    data, labels = iris_data.get_test_set()
    results = np.round(net.test(data))

    # ---------------------------
    # Statistics
    # ---------------------------
    # compute accuracy
    accuracy = compute_accuracy(results.T, labels.T)

    # report the result
    if accuracy > 0.9:
        print('\nSuccess!')
    else:
        print('\nFailed! (Network\'s accuracy is below 90%)')
    print('In this run, on the iris dataset, we achieve an accuracy of {} %'.format(str(accuracy * 100)))

    net.show()


class NeuralNetwork:

    def __init__(self):

        # list which will contain the losses by iteration after training
        self.loss = []
        # the layer providing the data (input layer)
        self.data_layer = None
        # the layer calculating the loss and the prediction
        self.loss_layer = None
        # the definition of the particular neural network
        self.layers = []

    # the forward pass of the network, returning activation of the last layer
    def forward(self, activation_tensor):

        # pass the input up the network
        for layer in self.layers:
            activation_tensor = layer.forward(activation_tensor)

        # return the activation of the last layer
        return self.loss_layer.forward(activation_tensor)

    # the raw backward pass during training
    def backward(self, label_tensor):

        # fetch the label from the data layer and pass it through the loss
        error_tensor = self.loss_layer.backward(label_tensor)

        # pass back the error recursively
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    # high level method to train the network
    def train(self, iterations):

        # iterate for a fixed number of steps
        for i in range(iterations):

            # get the training data
            input_tensor = self.data_layer.forward()
            label_tensor = self.data_layer.backward()

            # pass the input up the network
            self.forward(input_tensor)

            # calculate the loss of the network using the loss layer and save it
            estimated_loss = self.loss_layer.loss(label_tensor)
            self.loss.append(estimated_loss)

            # down the network including update of weights
            self.backward(label_tensor)

    # high level method to test a new input
    def test(self, input_tensor):

        return self.forward(input_tensor)

    # plot the loss curve over iterations
    def show(self):

        plt.plot(self.loss)
        plt.show()


class Data:

    def __init__(self, inputs, labels):

        self.data = {'train':
                     {'input': np.array(inputs[0]),
                      'label': np.array(labels[0])},
                     'test':
                     {'input': np.array(inputs[1]),
                      'label': np.array(labels[1])}
                     }
        self.n_train = len(inputs[0])
        self.n_test = len(inputs[1])

    def get_train_set(self):
        return self.data['train']['input'].T, self.data['train']['label'].T

    def get_test_set(self):
        return self.data['test']['input'].T, self.data['test']['label'].T


class IrisData(Data):

    train_per_test = 2
    categories = 3

    def __init__(self, do_shuffle=True):

        inputs = [[], []]
        labels = [[], []]
        samples_by_category = [[] for _ in range(self.categories)]

        r = IrisData.train_per_test / (IrisData.train_per_test + 1)

        data, target = load_iris(True)
        self.total = data.shape[0]
        self.feature_dim = data.shape[1]
        self.full_batch = 0

        for i in range(self.total):
            samples_by_category[target[i]].append(data[i, :])

        for i in range(self.categories):

            if do_shuffle:
                samples_by_category[i] = IrisData._get_shuffled_data(samples_by_category[i])

            n = len(samples_by_category[i])
            s = round(r * n)
            inputs[0] += samples_by_category[i][0:s]
            inputs[1] += samples_by_category[i][s:]
            labels[0] += [[int(j == i) for j in range(self.categories)]] * s
            labels[1] += [[int(j == i) for j in range(self.categories)]] * (n-s)

            self.full_batch += s

        super().__init__(inputs, labels)

    @staticmethod
    def _get_shuffled_data(data_as_list):

        index_list = list(range(len(data_as_list)))
        shuffle(index_list)
        return [data_as_list[i] for i in index_list]


class InputLayer:

    def __init__(self, data: Data):
        self.input_tensor, self.label_tensor = data.get_train_set()

    def forward(self):
        return np.copy(self.input_tensor)

    def backward(self):
        return np.copy(self.label_tensor)


def compute_accuracy(results, labels):
    print(results)

    correct = 0
    wrong = 0

    for column_results, column_labels in zip(results, labels):
        if column_results[column_labels > 0].all() > 0:
            correct += 1
        else:
            wrong += 1

    if correct == 0 and wrong == 0:
        return 0
    else:
        return correct / (correct + wrong)


if __name__ == '__main__':
    main()

