#!/usr/bin/env python3
"""
Inverse time decay in numpy
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy
    Args:
        alpha: is the original learning rate
        decay_rate: is the weight used to find the rate at which α will decay
        global_step: is the # of passes of gradient descent that have elapsed
        decay_step: is the number of passes of gradient descent that should
                    occur before alpha is decayed further
    Notes:
        The learning rate decay should occur in a stepwise fashion
    Return:
        The updated value for alpha
    """

    # inverse time decay
    # https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    # lr *= (1. / (1. decay rate * epoch)) * lr

    # But using the formula in tf.train.inverse_time_decay:
    # decayed_learning_rate =
    # learning_rate / (1 + decay_rate * floor(global_step / decay_step))
    # floor int()

    dr = decay_rate

    updated_dr = alpha / (1 + dr * int(global_step / decay_step))
    return updated_dr
