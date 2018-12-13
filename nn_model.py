import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


class model:
    def __init__(self):
        self.alpha = 0.01
        input_size = 9
        h1_out_size = 5
        o_out_size = 3
        self.weights_o = np.random.rand(h1_out_size+1, o_out_size,)*64
        self.weights_h1 = 0.02*np.random.rand(input_size+1, h1_out_size)-0.01

    def next_training_sample(self):
        # TODO fetch random input
        pass

    def forward(self, x, y, training=False):
        # input: x.shape (9,)
        # in_h1 weighed sum of input + bias (5,)
        # out_h1 sigmoid(in_h1) (5,)
        x_padded = np.pad(x, (1, 0), "constant", constant_values=1)
        in_h1 = np.dot(x_padded, self.weights_h1)
        out_h1 = sigmoid(in_h1)
        # Output layer
        # input: (6,)
        # in_o: (3,)
        # out_o: (3, )
        out_h1_padded = np.pad(out_h1, (1, 0), "constant", constant_values=1)
        in_o = np.dot(out_h1_padded, self.weights_o)
        out_o = in_o

        if training:
            error = (out_o - y)
            loss = 1/3*(np.sum(error**2))
            print(f"output:{out_o} y:{y} loss: {loss}")

            # Update the output layer's weights
            in_size, out_size = self.weights_o.shape
            g_prime_ink = np.ones((1, in_size))
            modified_error_k = (error).reshape(3, 1)
            modified_error_k = np.matmul(modified_error_k, g_prime_ink)
            gradient = 2*modified_error_k*out_h1_padded
            gradient = self.alpha*gradient.T
            new_weights_o = self.weights_o - gradient

            #Update the hidden layer's weights
            self.update_hidden_layer1(y, x, out_o)
            modified_error_j = np.dot(self.weights_o, modified_error_k)
            g_prime_inj = out_h1*(1-out_h1)


            self.weights_o = new_weights_o
        return out_o, loss

    def update_hidden_layer1(self, y, out_i, out_j):
        # TODO error back propagation
        pass

    def update_output_layer(self, y, out_j, out_k):
        in_size, out_size = self.weights_o.shape
        for k in range(out_size):
            #modified error for k
            modified_error = 2*(out_k[k]-y[k])
            modified_error *= out_j

            gradient = self.alpha*modified_error
            self.weights_o[:, k] -= gradient
