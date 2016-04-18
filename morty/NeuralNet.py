import numpy as np
import time

from numpy.random import randn
from random import shuffle


def heaviside(x): return (1 if x >= 0 else 0)


def d_heaviside(x): return (1 if x == 0 else 0)


def sigmoid(x): return 1 / (1 + np.exp(-x))


def d_sigmoid(x): return sigmoid(x) * (1 - sigmoid(x))


def MSE(y, y_est): return ((y - y_est) ** 2) / 2


def d_MSE(y, y_est): return -(y - y_est)


class NeuralNet:

    def __init__(self, sizes, activation_function=(sigmoid, d_sigmoid),
                 weights=None):
        # TODO: check whether bias, weight sizes are compatible with layer
        # sizes. Remember weights[layer][0] are the biases
        self.sizes = np.array(sizes)
        self.activation_function = activation_function[0]
        self.d_activation_function = activation_function[1]

        if weights is None:
            self.weights = [randn(y, x) for x, y in
                            zip(np.concatenate(([sizes[0]], sizes[:-1])) +
                            np.ones_like(sizes), sizes)]
        elif type(weights) is str:
            self.load_weights(weights)
        else:
            self.weights = np.array(weights)

    def __call__(self, network_input):
        return self.estimate(network_input)

    def estimate(self, network_input):
        return self.feed_forward(network_input)[0][-1]

    def feed_forward(self, network_input):
        # consider the input as output of the 0th layer
        layer_outputs = np.array([np.zeros(size) for size in self.sizes])
        local_fields = np.array([np.zeros(size) for size in self.sizes])

        for layer, layer_size in enumerate(self.sizes):
            # [1] is the "bias neuron"
            layer_input = np.concatenate(([1], layer_outputs[layer-1])) \
                          if layer != 0 \
                          else np.concatenate(([1], np.array(network_input)))

            for neuron in range(layer_size):
                local_weight = self.weights[layer][neuron]
                local_gradient = np.dot(layer_input, local_weight)
                local_fields[layer][neuron] = local_gradient
                local_output = self.activation_function(local_gradient)
                layer_outputs[layer][neuron] = local_output

        return (layer_outputs, local_fields)

    def train(self, data, labels, epochs=1, block_size=1, learn_rate=0.5):
        data = [np.array(datum) for datum in data]
        training_data = [list(i) for i in zip(data, labels)]
        training_size = len(training_data)

        for epoch in range(epochs):
            shuffle(training_data)
            blocks = [training_data[k:k+block_size]
                      for k in range(0, training_size, block_size)]

            for block_idx, block in enumerate(blocks):
                self.train_block(block, learn_rate)
            # print("Block %d/%d completed.\n" % (block_idx + 1, len(blocks)))

    def train_block(self, block, learn_rate):
        gradient = np.array([np.zeros_like(w) for w in self.weights])

        for data, label in block:
            gradient += self.back_propagation(data, label)

        # gradient /= len(block)
        self.weights -= learn_rate * gradient

    def back_propagation(self, data, label):
        (layer_outputs, local_fields) = self.feed_forward(data)
        layer_outputs = [np.concatenate(([1], layer_output))
                         for layer_output in layer_outputs]
        layer_inputs = [np.concatenate(([1], data))] + layer_outputs[:-1]
        network_output = self.estimate(data)

        d_outputs = [self.d_activation_function(local_field)
                     for local_field in local_fields]
        local_gradient = np.array([np.zeros(int(size)) for size in self.sizes])

        for layer, layer_size in reversed(list(enumerate(self.sizes))):
            for neuron in range(layer_size):
                # print("n%d of l%d\n" % (neuron, layer))
                if layer == len(self.sizes) - 1:
                    mse = d_MSE(label[neuron], network_output[neuron])
                    d_out = d_outputs[layer][neuron]
                    local_gradient[layer][neuron] = d_out * mse

                else:
                    d_out = d_outputs[layer][neuron]
                    local_gradient[layer][neuron] = d_out * \
                        sum([self.weights[layer + 1][w][neuron] *
                            local_gradient[layer + 1][w]
                            for w in range(self.sizes[layer + 1])])

        return [np.multiply(*np.meshgrid(layer_inputs[i], local_gradient[i]))
                for i in range(len(self.sizes))]

    def validate(self, data, labels):
        data = [np.array(datum) for datum in data]
        training_data = [list(i) for i in zip(data, labels)]
        training_size = len(training_data)

        confusion_matrix = np.zeros((10, 10))
        accuracy = 0

        for sample, label in training_data:
            y_est = np.argmax(self.estimate(sample))
            y = np.argmax(label)
            confusion_matrix[y][y_est] += 1
            if y_est == y:
                accuracy += 1

        np.set_printoptions(suppress=True)
        print(confusion_matrix)
        return accuracy / training_size

    def save_weights(self, filename="weights", timestamp=True):
        if timestamp:
            filename = time.strftime("%Y%m%d-%H%M%S-") + filename
        filename = "./Weights/" + filename + ".npy"

        np.save(filename, np.array(self.weights))

    def load_weights(self, filename=None):
        if filename is not None:
            filename = "./Weights/" + filename + ".npy"
            self.weights = np.load(filename)
        raise ValueError('No filename given.')
