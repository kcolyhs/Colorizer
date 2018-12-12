import numpy as np
from random import random


class model:
    """
    """
    def __init__(self, output_size = 3):
        self.alpha = 0.01
        self.outputl = [nn_node.random_init(9) for x in range(0, output_size)]
        self.hidden1 = [nn_node.random_init(5) for x in range(0, 9)]
        self.weights_o = np.random.rand(output_size,9)
        self.bias_o = np.random.rand(output_size)
        self.weights_h1 = np.random.rand(9, 9)

    def next_training_sample(self):
        # TODO fetch random input
        pass

    def forward(self, input):
        # Check input
        

    def loss(output, actual):
        error = (np.sum(output-actual)**2)
        return error

    def update_hidden_layer(self, layer):
        pass

    def update_output_layer(self, loss):

        for node in self.outputl:
            for j in range(node.input_size):
                in_k = node.weighted_sum
                out_k = node.out
                gradient = - self.alpha*(out_k*(1-out_k))  # * ?out_k
                node.weights[j] -= gradient
            # node.bias = node.weight - self.alpha*(loss*(1-loss))


class nn_node:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.input_size = 0
        self.inputs = []
        self.out
        self.weighted_sum

    @staticmethod
    def random_init(n):
        node = nn_node()
        node.input_size = n
        node.weight = [random() for _ in range(node.input_size)]
        node.bias = random()
        return node

    def get_output(self, inputs):
        y = inputs * self.weights + self.bias
        y = np.sum(y)
        output = 1.0 / (1 + np.exp(-y))
        return output
