#!/usr/bin/env python3
"""
Creates a confusion matrix
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix
    Args:
        - labels: is a one-hot numpy.ndarray of shape (m, classes) containing
                  the correct labels for each data point
                  m: is the number of data points
                  classes:  is the number of classes
        - logits: is a one-hot numpy.ndarray of shape (m, classes) containing
                  the predicted labels
    Return:
        A confusion numpy.ndarray of shape (classes, classes) with row indices
        representing the correct labels and column indices representing the
        predicted labels
    """
    # print("Shape of labels: {}".format(labels.shape))
    # print("Shape of logits: {}".format(logits.shape))

    c_matrix = np.matmul(labels.T, logits)

    return c_matrix
