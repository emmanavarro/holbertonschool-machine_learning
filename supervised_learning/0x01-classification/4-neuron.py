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

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron """
        # z = W.X + b
        z = np.matmul(self.W, X) + self.b
        sigmoid_func = 1 / (1 + np.exp(-z))
        self.__A = sigmoid_func
        return self.__A

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions """
        self.forward_prop(X)
        A = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return A, cost
