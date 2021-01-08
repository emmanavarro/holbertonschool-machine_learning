#!/usr/bin/env python3
"""
Contains a function to create the forward propagation graph for the neural
network
"""

import tensorflow as tf


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.
    Args:
        - x: is the placeholder for the input data.
        - layer_sizes: is a list containing the number of nodes in each layer
          of the network.
        - activations: is a list containing the activation functions for each
          layer of the network.
    Return:
        The prediction of the network in tensor form.
    """
    prediction = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        prediction = create_layer(prediction, layer_sizes[i], activations[i])
    return prediction
