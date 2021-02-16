#!/usr/bin/env python3
""" Contains the inception_block function """

import tensorflow.keras as Keras


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in "Going Deeper with
    Convolutions (2014)":
    Args:
        - A_prev output from the previous layer
        - filters tuple or list containing F1, F3R, F3,F5R, F5, FPP,
          respectively:
            F1: number of filters in the 1x1 convolution
            F3R: number of filters in the 1x1 convolution before the 3x3
                 convolution
            F3: number of filters in the 3x3 convolution
            F5R: number of filters in the 1x1 convolution before the 5x5
                 convolution
            F5: number of filters in the 5x5 convolution
            FPP: number of filters in the 1x1 convolution after the max pooling
    Return: concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # He et. al initialization for the layers weights
    initializer = Keras.initializers.he_normal(seed=None)

    # Conv 1x1
    my_layer = Keras.layers.Conv2D(filters=F1,
                                   kernel_size=(1, 1),
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer=initializer,
                                   )(A_prev)

    # Conv 1x1 before the 3x3 convolution
    my_layer1 = Keras.layers.Conv2D(filters=F3R,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=initializer,
                                    )(A_prev)

    # Conv 3x3 convolution
    my_layer1 = Keras.layers.Conv2D(filters=F3,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=initializer,
                                    )(my_layer1)

    # Conv 1x1 before the 5x5 convolution
    my_layer2 = Keras.layers.Conv2D(filters=F5R,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=initializer,
                                    )(A_prev)

    # Conv 5x5
    my_layer2 = Keras.layers.Conv2D(filters=F5,
                                    kernel_size=(5, 5),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=initializer,
                                    )(my_layer2)

    # Max pooling layer with kernels of shape 3x3 with 1x1 strides
    my_layer3 = Keras.layers.MaxPool2D(pool_size=(3, 3),
                                       padding='same',
                                       strides=(1, 1)
                                       )(A_prev)

    # Convolutional layer 1x1 convolution after the max pooling
    my_layer3 = Keras.layers.Conv2D(filters=FPP,
                                    kernel_size=(1, 1),
                                    padding='same',
                                    activation='relu',
                                    kernel_initializer=initializer,
                                    )(my_layer3)

    output = Keras.layers.concatenate([my_layer, my_layer1,
                                      my_layer2, my_layer3])

    return output
