#!/usr/bin/env python3
""" Contains the resnet50 function """

import tensorflow.keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture

    Notes:
        - You can assume the input data will have shape (224, 224, 3)
        - All convolutions inside and outside the blocks should be followed by
          batch normalization along the channels axis and a rectified linear
          activation (ReLU), respectively.
        - All weights should use he normal initialization

    Return: the keras model
    """
    # He et. al initialization for the layers weights
    initializer = K.initializers.he_normal(seed=None)

    X = K.Input(shape=(224, 224, 3))

    # Conv 7x7 + 2(S)
    my_layer = K.layers.Conv2D(filters=64,
                               kernel_size=(7, 7),
                               strides=(2, 2),
                               padding='same',
                               kernel_initializer=initializer,
                               )(X)

    my_layer = K.layers.BatchNormalization(axis=3)(my_layer)
    my_layer = K.layers.Activation('relu')(my_layer)

    # MaxPool 3x3 + 2(S)
    my_layer = K.layers.MaxPool2D(pool_size=(3, 3),
                                  padding='same',
                                  strides=(2, 2))(my_layer)

    my_layer = projection_block(my_layer, [64, 64, 256], 1)
    for i in range(2):
        my_layer = identity_block(my_layer, [64, 64, 256])

    my_layer = projection_block(my_layer, [128, 128, 512])
    for i in range(3):
        my_layer = identity_block(my_layer, [128, 128, 512])

    my_layer = projection_block(my_layer, [256, 256, 1024])
    for i in range(5):
        my_layer = identity_block(my_layer, [256, 256, 1024])

    my_layer = projection_block(my_layer, [512, 512, 2048])
    for i in range(2):
        my_layer = identity_block(my_layer, [512, 512, 2048])

    # Avg pooling layer with kernels of shape 7x7
    my_layer = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         padding='same')(my_layer)

    # Fully connected softmax output layer with 1000 nodes
    my_layer = K.layers.Dense(units=1000,
                              activation='softmax',
                              kernel_initializer=initializer,
                              )(my_layer)

    model = K.models.Model(inputs=X, outputs=my_layer)

    return model
