#!/usr/bin/env python3
"""
Normalization Constants
"""

import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix:
    Arg:
        - X is the numpy.ndarray of shape (m, nx) to normalize
            - m is the number of data points
            - nx is the number of features
    Return:
        The mean and standard deviation of each feature, respectively
    """
    # Mean
    mu = np.mean(X, axis=0)
    # Standard deviation
    sigma = np.std(X, axis=0)

    return mu, sigma
