#!/usr/bin/env python3
"""
Contains a function to create the training operation for the network
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates the training operation for the network.
    Args:
        - loss: is the loss of the network’s prediction.
        - alpha: is the learning rate.
    Return:
        An operation that trains the network using gradient descent.
    """
    operation = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return operation
