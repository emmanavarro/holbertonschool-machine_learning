#!/usr/bin/env python3
"""
Module used to
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm:
    Args:
        alpha:   is the learning rate
        beta2:   is the momentum weight
        epsilon: is a small number to avoid division by zero
        var:     is a numpy.ndarray containing the variable to be updated
        grad:    is a numpy.ndarray containing the gradient of var
        s:       is the previous second moment of var
    Return:
        The updated variable and the new moment, respectively
    """

    dw = grad
    w = var

    s_new = beta2 * s + (1 - beta2) * (dw ** 2)
    W = w - alpha * (dw / ((s_new ** 0.5) + epsilon))

    return W, s_new
