#!/usr/bin/env python3
"""
RMSProp
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Updates a variable using the RMSProp optimization algorithm:
    Args:
        loss     is the loss of the network
        alpha:   is the learning rate
        beta2:   is the RMSProp weight
        epsilon: is a small number to avoid division by zero
    Return:
        The RMSProp optimization operation
    """
    # tf.train.RMSPropOptimizer
    # https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/train/RMSPropOptimizer.md
    # Optimizer that implements the RMSProp algorithm.

    train = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                      decay=beta2,
                                      epsilon=epsilon)

    op = train.minimize(loss)
    return op
