#!/usr/bin/env python3
"""
Method to create a layer in a neural network
"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    Creates a layer in a neural network.
    Args:
        - prev: is the tensor output of the previous layer.
        - n: is the number of nodes in the layer to create.
        - activation: is the activation function that the layer should use:
            - tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG") is
              used to implement He et. al initialization for the layer weights.
            - each layer should be given the name "layer".
    Return:
        The tensor output of the layer.
    """
    raw_layer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    output_tensor = tf.layers.Dense(units=n,
                                    activation=activation,
                                    kernel_initializer=raw_layer,
                                    name='layer')

    return output_tensor(prev)
