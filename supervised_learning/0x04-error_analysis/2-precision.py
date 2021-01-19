#!/usr/bin/env python3
"""
Calculates the precision in a confusion matrix
"""

import numpy as np


def precision(confusion):
    """
    calculates the precision for each class in a confusion matrix
    Args:
    - confusion: is a confusion numpy.ndarray of shape (classes, classes) where
      row indices represent the correct labels and column indices represent the
      predicted labels
        * classes: is the number of classes
    Return:
    A numpy.ndarray of shape (classes,) containing the precision of each class
    """
    true_positive = np.diagonal(confusion)
    all_positive = np.sum(confusion, axis=0)
    precision = true_positive / all_positive

    return precision
