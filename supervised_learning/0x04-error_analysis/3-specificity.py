#!/usr/bin/env python3
"""
Calculates the specificity in a confusion matrix
"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix
    Arg:
    - confusion: is a confusion numpy.ndarray of shape (classes, classes) where
      row indices represent the correct labels and column indices represent the
      predicted labels
        * classes: is the number of classes
    Return:
    A numpy.ndarray of shape (classes,) containing the specificity of each
    class
    """
    # tp: true positive, fp: false positive, tn: true negative,
    # fn: false negative
    tp = np.diagonal(confusion)
    fn = np.sum(confusion, axis=1) - tp
    fp = np.sum(confusion, axis=0) - tp
    tn = np.sum(confusion) - (tp + fn + fp)
    specificity = tn / (tn + fp)

    return specificity
