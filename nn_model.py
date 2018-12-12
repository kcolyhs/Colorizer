import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class model:
    def __init__(self):
        self.alpha = 0.01
        input_size = 9
        h1_out_size = 5
        o_out_size = 3
        self.weights_o = np.random.rand(h1_out_size, o_out_size,)
        self.bias_o = np.random.rand(o_out_size)
        self.weights_h1 = np.random.rand(input_size, h1_out_size)
        self.bias_h1 = np.random.rand(h1_out_size)

    def next_training_sample(self):
        # TODO fetch random input
        pass

    def forward(self, x, y, training=False):
        # input x (9,)
        out_h1 = np.dot(x, self.weights_h1)
        out_h1 += self.bias_h1
        out_h1 = sigmoid(out_h1)
        out_o = np.dot(out_h1, self.weights_o)
        out_o += self.bias_o
        out_o = sigmoid(out_o)

        if training:
            loss = (np.sum(out_o - y)**2)
            self.update_output_layer(loss, out_h1, out_o)
            self.update_hidden_layer1(loss, x, out_o)

        return out_o

    def update_hidden_layer1(self, layer, in_j, out_j):
        # TODO error back propagation
        pass

    def update_output_layer(self, loss, in_k, out_k):
        in_size, out_size = self.weights_o.shape
        for k in range(out_size):
            gradient = self.alpha*(out_k*(1-out_k))*out_k  # * ?out_k
            self.weights_o[k] -= gradient
            # node.bias = node.weight - self.alpha*(loss*(1-loss))
