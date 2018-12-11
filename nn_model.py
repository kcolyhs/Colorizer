import numpy as np


class model:
    """
    """
    def __init__(self):
        self.input
        self.output = []
        self.hidden1 = [nn_node.random_init() for x in range(0,10)]
        self.alpha = 0.01


    def next_training_sample():
        pass

    def evaluate_gradient():
        pass

    def loss(output, actual):
        error = (np.sum(output-actual)**2)
        return error

    def error_backpropagation(layer):
        pass

    def update_output_weights(loss):
        for node in output:
            node.weight = node.weight -alpha*(loss*(1-loss))
            node.bias =


class nn_node:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.input_size = 0

    def random_init():
        @staticmethod
        node = nn_node()
        node.weight = [random() for _ in range(self.input_size)]
        node.bias = random()
        return node

    def get_output(self.inputs):
        y = inputs * self.weights + self.bias
        y = np.sum(y)
        output = 1.0 / (1 + np.exp(-y))
        return output