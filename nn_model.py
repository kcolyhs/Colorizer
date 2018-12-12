import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class model:
    def __init__(self):
        self.alpha = 0.1
        input_size = 9
        h1_out_size = 5
        o_out_size = 3
        self.weights_o = np.random.rand(h1_out_size+1, o_out_size,)*0.01
        self.weights_h1 = np.random.rand(input_size+1, h1_out_size)*0.01

    def next_training_sample(self):
        # TODO fetch random input
        pass

    def forward(self, x, y, training=False):
        # input x (9+1,)
        in_h1 = np.pad(x, (1, 0), "constant", constant_values=1)
        out_h1 = np.dot(in_h1, self.weights_h1)
        out_h1 = sigmoid(out_h1)
        in_o = np.pad(out_h1, (1, 0), "constant", constant_values=1)
        out_o = np.dot(in_o, self.weights_o)
        out_o = sigmoid(out_o) * 256

        if training:
            error = (out_o - y)
            loss = 1/3*(np.sum(error**2))
            print(f"output:{out_o} y:{y} loss: {loss}")
            self.update_output_layer(y, in_o, out_o)
            self.update_hidden_layer1(y, x, out_o)

        return out_o, loss

    def update_hidden_layer1(self, y, in_j, out_j):
        # TODO error back propagation
        pass

    def update_output_layer(self, y, out_j, out_k):
        in_size, out_size = self.weights_o.shape
        for k in range(out_size):
            modified_error = 2*(out_k[k]-y[k])
            modified_error = np.multiply(modified_error, out_j)
            modified_error = np.multiply(modified_error, (1-out_j))
            modified_error = np.multiply(modified_error, out_j)

            gradient = self.alpha*modified_error
            self.weights_o[:, k] -= gradient
