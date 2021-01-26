#!/usr/bin/env python3
"""
One-hot matrix
"""

import tensorflow.keras as keras


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix
    The last dimension of the one-hot matrix must be the number of classes
    Returns:
        The one-hot matrix
    """
    one_hot_matrix = keras.utils.to_categorical(labels, classes)

    return one_hot_matrix
