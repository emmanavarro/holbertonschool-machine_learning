#!/usr/bin/env python3
"""
Contains a function to calculate the accuracy of a prediction
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.
    Args:
        - y: is a placeholder for the labels of the input data.
        - y_pred: is a tensor containing the network’s predictions.
    Return:
        A tensor containing the decimal accuracy of the prediction.
    """
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy
