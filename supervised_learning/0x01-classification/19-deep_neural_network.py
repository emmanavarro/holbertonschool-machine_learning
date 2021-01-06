#!/usr/bin/env python3
"""
Deep neural network
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification
    """
    def __init__(self, nx, layers):
        """
        Class constructor
        Args:
            - nx: is the number of input features
            - layers: is a list representing the number of nodes in each layer
              of the network
        Public instance attributes:
            - L: The number of layers in the neural network.
            - cache: A dictionary to hold all intermediary values of the
              network.
            - weights: A dictionary to hold all weights and biased of the
              network.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for lay in range(self.__L):
            if not isinstance(layers[lay], int) or layers[lay] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if lay == 0:
                # He et al. initialization for weights in first layer
                He = (np.random.randn(layers[lay], nx)
                      * np.sqrt(2 / nx))
                self.__weights["W{}".format(lay + 1)] = He
            else:
                # He et al. initialization for weights
                He = (np.random.randn(layers[lay], layers[lay - 1])
                      * np.sqrt(2 / layers[lay - 1]))
                self.__weights["W{}".format(lay + 1)] = He
            # Zero initialization for biases
            self.__weights["b{}".format(lay + 1)] = np.zeros((layers[lay], 1))

    @property
    def L(self):
        """ Getter method for __L """
        return self.__L

    @property
    def cache(self):
        """ Getter method for __cache """
        return self.__cache

    @property
    def weights(self):
        """ Getter method for __weights """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network, updates the
        private attribute __cache.
        - X: is a numpy.ndarray with shape (nx, m) that contains the input data
            - nx: is the number of input features to the neuron
            - m: is the number of examples
        Return:
            The output of the neural network and the cache, respectively
        """
        self.__cache["A0"] = X
        for layer in range(self.__L):
            weights = self.__weights["W{}".format(layer + 1)]
            a_ = self.__cache["A{}".format(layer)]
            biases = self.__weights["b{}".format(layer + 1)]

            # Forward propagation function
            Z = np.matmul(weights, a_) + biases

            # Sigmoid activation function
            sigmoid = 1 / (1 + np.exp(-Z))

            # Update cache
            self.__cache["A{}".format(layer + 1)] = sigmoid

        return self.__cache["A{}".format(layer + 1)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        Args:
            - Y: is a numpy.ndarray with shape (1, m) that contains the correct
                 labels for the input data.
            - A: is a numpy.ndarray with shape (1, m) containing the activated
                 output of the neuron for each example.
        To avoid division by zero errors, it is used 1.0000001 - A instead of
        1 - A.
        Return:
            The cost of the model.
        """
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost
