#!/usr/bin/env python3
"""
Optimize
"""

import tensorflow.keras as keras


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a keras model with categorical crossentropy
    loss and accuracy metrics
    Args:
        - network: is the model to optimize
        - alpha: is the learning rate
        - beta1: is the first Adam optimization parameter
        - beta2: is the second Adam optimization parameter
    Returns: None
    """
    optimizer = keras.optimizers.Adam(alpha, beta1, beta2)
    network.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])

    return None
