import numpy as np
from random import random


class model:
    def __init__(self):
        self.alpha = 0.01
        input_size = 9
        h1_out_size = 5
        o_out_size = 3
        self.weights_o = np.random.rand(o_out_size,h1_out_size)
        self.bias_o = np.random.rand(o_out_size)
        self.weights_h1 = np.random.rand(h1_out_size, input_size)
        self.bias_h1 = np.random.rand(o_out_size)

    def next_training_sample(self):
        # TODO fetch random input
        pass

    def forward(self, input):
        flat_input = input.flatten()



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
                gradient = - self.alpha*(out_k*(1-out_k))*out_k  # * ?out_k
                node.weights[j] -= gradient
            # node.bias = node.weight - self.alpha*(loss*(1-loss))


class nn_node:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.input_size = 0
        self.inputs = []
        self.out = None
        self.weighted_sum = None

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
