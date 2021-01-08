#!/usr/bin/env python3
"""
Contains a function to calculate the softmax cross-entropy loss of a prediction
"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculate the softmax cross-entropy loss of a prediction.
    Args:
        - y: is a placeholder for the labels of the input data.
        - y_pred: is a tensor containing the network’s predictions.
    Return:
        A tensor containing the loss of the prediction.
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
