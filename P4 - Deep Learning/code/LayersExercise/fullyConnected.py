import numpy as np


class FullyConnected:

    def __init__(self, input_size, output_size, batch_size, delta):

        # initialize the weights randomly
        self.weights = np.random.rand(output_size, input_size + 1)

        # the current activations have to be stored to be accessible in the back propagation step
        self.activations = np.zeros((input_size + 1, batch_size))  # "pre-allocation"

        # allow individual learning rates per hidden layer
        self.delta = delta

    def forward(self, input_tensor):

        # put together the activations from the input_tensor
        # add an additional row of ones to include the bias (such that w^T * x + b becomes w^T * x equivalently)
        # TODO
        biases = np.ones((np.shape(input_tensor)[1], 1))
        self.input_x = np.vstack((input_tensor, biases.transpose()))

        # perform the forward pass just by matrix multiplication
        layer_output = np.dot(self.weights, self.input_x)  # TODO
        # print(layer_output.shape)
        return layer_output

    def backward(self, error_tensor):

        # update the layer using the learning rate and E * X^T,
        # where E is the error from higher layers and X are the activations stored from the forward pass
        #
        # 1. calculate the error for the next layers using the transposed weights and the error
        # TODO
        # print(error_tensor.shape)
        # print(self.weights[:, np.shape(self.weights[0]-1)].shape)

        En_x = np.dot(self.weights[:, :np.shape(self.weights)[1]-1].transpose(), error_tensor)

        # 2. update this layer's weights
        # TODO
        self.gradient_weights = np.dot(error_tensor, self.input_x.transpose())

        self.weights = self.weights - self.delta * self.gradient_weights

        # the bias of this layer does not affect the layers before, so delete it from the return value
        error_tensor_new = En_x # TODO
        return error_tensor_new
