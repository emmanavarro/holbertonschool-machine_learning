#!/usr/bin/env python3
"""
Sequential
"""

import tensorflow.keras as keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library
    Args:
        - nx: is the number of input features to the network
        - layers: is a list containing the number of nodes in each layer of the
          network
        - activations: is a list containing the activation functions used for
          each layer of the network
        - lambtha: is the L2 regularization parameter
        - keep_prob: is the probability that a node will be kept for dropout

        Note: You are not allowed to use the Input class
    Returns:
        The keras model
    """
    model = keras.Sequential()
    regularizer = keras.regularizers.l2(lambtha)

    # creating the first densely-connected layer
    model.add(keras.layers.Dense(units=layers[0],
                                 activation=activations[0],
                                 kernel_regularizer=regularizer,
                                 input_shape=(nx,)))

    # creating the subsequent densley-connected layers
    for i in range(1, len(layers)):
        model.add(keras.layers.Dropout(1 - keep_prob))
        model.add(keras.layers.Dense(units=layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=regularizer))

    return model
