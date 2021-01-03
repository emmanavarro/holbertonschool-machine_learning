#!/usr/bin/env python3
""" Defines a single neuron performing binary classification """

import numpy as np


class Neuron:
    """ Neuron class """
    def __init__(self, nx):
        """ Class constructor """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Getter method for W """
        return self.__W

    @property
    def b(self):
        """ Getter method for b """
        return self.__b

    @property
    def A(self):
        """ Getter method for A """
        return self.__A
