#!/usr/bin/env python3
"""
L2 Regularization Cost
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    calculates the cost of a neural network with L2 regularization
    Args:
        - cost is the cost of the network without L2 regularization
        - lambtha is the regularization parameter
        - weights is a dictionary of the weights and biases (numpy.ndarrays) of
          the neural network
        - L is the number of layers in the neural network
        - m is the number of data points used
    Returns: the cost of the network accounting for L2 regularization
    """
    # from: https://bit.ly/2IAtnFN
    # Cost function = Loss + (lambtha / 2 * m) *  Σ | w | ^ 2

    if (L == 0):
        return 0

    lambtha
    sum_w = 0

    for keys in weights:
        if (keys[0] == "W"):    # selecting only weights from the dict
            values = weights[keys]
            sum_w += np.linalg.norm(values)

    cost_l2 = cost + (lambtha / (2 * m)) * sum_w
    return(cost_l2)
