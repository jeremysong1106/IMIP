import numpy as np


class SoftMax:

    def __init__(self, categories, batch_size):

        # the current activations have to be stored to be accessible in the back propagation step
        self.activations = np.zeros((categories, batch_size))  # "pre-allocation"

    def forward(self, input_tensor):
        # print(input_tensor.shape)

        # store the activations from the input_tensor
        self.activations = np.copy(input_tensor)
        x_max = np.max(input_tensor, axis=0)
        x_max = np.tile(x_max, (input_tensor.shape[0], 1))
        # print(x_max)
        X = input_tensor - x_max
        X = np.exp(X)
        sum = np.sum(X, axis=0)
        sum_mat = np.tile(sum, (input_tensor.shape[0], 1))
        # print(sum_mat.shape)
        self.activations = X / sum_mat

        # apply SoftMax to the scores: e(x_i) / sum(e(x))
        # TODO
        # ...
        # print(self.activations.shape)
        return self.activations

    def backward(self, label_tensor):

        error_tensor = np.copy(self.activations)

        #  Given:
        #  - the labels are one-hot vectors
        #  - the loss is cross-entropy (as implemented below)
        # Idea:
        # - decrease the output everywhere except at the position where the label is correct
        # - implemented by increasing the output at the position of the correct label
        # Hint:
        # - do not let yourself get confused by the terms 'increase/decrease'
        # - instead consider the effect of the loss and the signs used for the backward pass

        # TODO
        label_tensor[label_tensor == 1] = -1
        error_tensor += label_tensor
        # ...
        # print(error_tensor.shape)
        return error_tensor

    def loss(self, label_tensor):

        loss = 0

        # iterate over all elements of the batch and sum the loss
        # TODO
        # ... # loss is the negative log of the activation of the correct position
        self.y_hat = self.activations * label_tensor
        # print(self.y_hat)
        y = np.sum(self.activations * label_tensor, axis=0)  # sum by row
        loss = np.sum(-np.log(y + np.finfo(float).eps))
        # print(loss)
        # print(loss)
        return loss
