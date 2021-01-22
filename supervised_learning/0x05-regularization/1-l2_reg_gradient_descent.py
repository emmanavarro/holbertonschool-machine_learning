#!/usr/bin/env python3
"""
Gradient Descent with L2 Regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Function that updates the weights and biases of a neural network using
    gradient descent with L2 regularization:
    Args:
    - Y: one-hot numpy.ndarray of shape (classes, m) that contains the
          correct labels for the data
          - classes is the number of classes
          - m is the number of data points
    - weights: dictionary of the weights and biases of the neural network
    - cache: dictionary of the outputs of each layer of the neural network
    - alpha: learning rate
    - lambtha: L2 regularization parameter
    - L: number of layers of the network
    Note: The NN uses tanh activations on each layer except the last, which
          uses a softmax activation
          The weights and biases of the network should be updated in place
    """

    weights2 = weights.copy()
    m = Y.shape[1]

    for n in reversed(range(L)):

        if n == L - 1:
            dz = cache["A{}".format(n + 1)] - Y
            dw = np.matmul(dz, cache["A{}".format(n)].T) / m
        else:
            dz1 = np.matmul(weights2["W{}".format(n + 2)].T, dz)
            dz2 = 1 - (cache["A{}".format(n + 1)] ** 2)
            dz = dz1 * dz2
            dw = np.matmul(dz, cache["A{}".format(n)].T) / m

        dw_reg = dw + (lambtha / m) * weights2["W{}".format(n + 1)]
        db = np.sum(dz, axis=1, keepdims=True) / m

        weights["W{}".format(n + 1)] = weights["W{}".format(n + 1)] \
            - (alpha * dw_reg)
        weights["b{}".format(n + 1)] = weights["b{}".format(n + 1)] \
            - (alpha * db)
