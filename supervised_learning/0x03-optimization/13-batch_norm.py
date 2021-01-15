#!/usr/bin/env python3
"""
Batch Normalization
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a NN using batch normalization:
    Args:
        Z: is a numpy.nd array of shape (m, n) that should be normalized
            - m is the number of data points
            - n is the number of features in Z
        gamma: is a numpy.ndarray of shape (1, n) containing the scales used
                for batch normalization
        beta: is a numpy.ndarray of shape (1, n) containing the offsets used
              for batch normalization
        epsilon: is a small number used to avoid division by zero
    Return:
        The the normalized Z matrix
    """

    # https://www.youtube.com/watch?v=tNIpEZLv_eg

    # mean
    mu = Z.mean(0)
    # std deviation
    std_dev = Z.std(0)
    # variance
    var = Z.std(0) ** 2
    # z normalized
    z_normalized = (Z - mu) / ((var + epsilon) ** (0.5))
    # We dont want all the units to have mean 0 and variance 1
    Z_adjust = gamma * z_normalized + beta

    return Z_adjust
