#!/usr/bin/env python3
"""
Calculates F1 score of a confusion matrix
"""

import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix
    Arg:
        - confusion: is a confusion numpy.ndarray of shape (classes, classes)
          where row indices represent the correct labels and column indices
          represent the predicted labels
            * classes: is the number of classes
    Return:
    numpy.ndarray of shape (classes,) containing the F1 score of each class
    """
    # s: sensitivity, p: precision
    s = sensitivity(confusion)
    p = precision(confusion)
    F_1_score = 2 * ((p * s) / (p + s))

    return F_1_score
