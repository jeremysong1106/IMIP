import numpy as np


class ReLU:

    def __init__(self, input_size, batch_size):

        # the current activations have to be stored to be accessible in the back propagation step
        self.activations = np.zeros((input_size, batch_size))  # "pre-allocation"

    def forward(self, input_tensor):

        # store the activations from the input_tensor
        # TODO
        input_tensor[input_tensor <= 0] = 0
        self.activations = input_tensor



        # the output is max(0, activation)
        layer_output = self.activations # TODO
        # print(layer_output.shape)
        return layer_output

    def backward(self, error_tensor):

        # the gradient is zero whenever the activation is negative
        self.activations[self.activations > 0] = 1
        # print(error_tensor.shape)
        return error_tensor * self.activations
