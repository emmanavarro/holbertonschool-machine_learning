#!/usr/bin/env python3
"""
Input
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

        Note: You are not allowed to use the Sequential class
    Returns:
        The keras model
    """
    inputs = keras.Input(shape=(nx,))
    regularizer = keras.regularizers.l2(lambtha)

    # Input layer creation
    outputs = keras.layers.Dense(units=layers[0],
                                     kernel_regularizer=regularizer,
                                     activation=activations[0],
                                     input_shape=(nx,))(inputs)

    # Subsequent layer creation
    for i in range(1, len(layers)):
        outputs = keras.layers.Dropout(1 - keep_prob)(outputs)
        outputs = keras.layers.Dense(units=layers[i],
                                          activation=activations[i],
                                          kernel_regularizer=regularizer
                                          )(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
