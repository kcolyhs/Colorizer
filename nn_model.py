import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))
np.random.seed(1)

class model:
    def __init__(self):
        self.alpha = 0.01
        input_size = 9
        h_out_size = 100
        o_out_size = 3
        self.weights_o = np.random.rand(h_out_size+1, o_out_size,)*64
        self.weights_h = 0.02*np.random.rand(input_size+1, h_out_size)-0.01

    def next_training_sample(self):
        # TODO fetch random input
        pass

    def forward(self, x, y, training=False):
        # Hidden  layer
        x_pad = np.pad(x, (1, 0), "constant", constant_values=1)
        in_h = np.dot(x_pad, self.weights_h)
        out_h = sigmoid(in_h)
        # Output layer
        out_h_pad = np.pad(out_h, (1, 0), "constant", constant_values=1)
        in_o = np.dot(out_h_pad, self.weights_o)
        out_o = in_o

        if training:
            error = (out_o - y)
            loss = (np.sum(error**2))
            print(f"output:{out_o} y:{y} loss: {loss}")

            # Update the output layer's weights
            mod_errk = error
            gradk = 2*self.alpha*np.matmul(out_h_pad.reshape(101,1),
                                              mod_errk.reshape(1,3))
            new_weights_o = self.weights_o - gradk

            #Update the hidden layer's weights (10,5)

            g_prime_inj = out_h_pad*(1-out_h_pad)
            mod_errj = np.dot(self.weights_o, mod_errk)
            mod_errj *=  g_prime_inj
            gradj = 2*self.alpha*np.matmul(x_pad.reshape(10,1),
                                              mod_errj[1:].reshape(1,100))
            self.weights_h - gradj

            self.weights_o = new_weights_o
        return out_o, loss
