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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for lay in range(self.L):
            if not isinstance(layers[lay], int) or layers[lay] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if lay == 0:
                # He et al. initialization for weights in first layer
                He = (np.random.randn(layers[lay], nx)
                      * np.sqrt(2 / nx))
                self.weights["W{}".format(lay + 1)] = He
            else:
                # He et al. initialization for weights
                He = (np.random.randn(layers[lay], layers[lay - 1])
                      * np.sqrt(2 / layers[lay - 1]))
                self.weights["W{}".format(lay + 1)] = He
            # Zero initialization for biases
            self.weights["b{}".format(lay + 1)] = np.zeros((layers[lay], 1))
