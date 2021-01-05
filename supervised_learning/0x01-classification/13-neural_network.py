#!/usr/bin/env python3
""" Defines a neural network """

import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer performing binary
       classification
    """
    def __init__(self, nx, nodes):
        """ Class constructor
            Args:
                - nx: is the number of input features.
                - nodes: is the number of nodes found in the hidden layer.
            Public instance attributes:
                - W1: The weights vector for the hidden layer. Upon
                  instantiation, it should be initialized using a random
                  normal distribution.
                - b1: The bias for the hidden layer. Upon instantiation, it
                  should be initialized with 0’s.
                - A1: The activated output for the hidden layer. Upon
                  instantiation, it should be initialized to 0.
                - W2: The weights vector for the output neuron. Upon
                  instantiation, it should be initialized using a random
                  normal distribution.
                - b2: The bias for the output neuron. Upon instantiation, it
                  should be initialized to 0.
                - A2: The activated output for the output neuron (prediction).
                  Upon instantiation, it should be initialized to 0.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ Getter method for W1 """
        return self.__W1

    @property
    def b1(self):
        """ Getter method for b1 """
        return self.__b1

    @property
    def A1(self):
        """ Getter method for A1 """
        return self.__A1

    @property
    def W2(self):
        """ Getter method for W2 """
        return self.__W2

    @property
    def b2(self):
        """ Getter method for b2 """
        return self.__b2

    @property
    def A2(self):
        """ Getter method for A2 """
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        Args:
            - X: is a numpy.ndarray with shape (nx, m) that contains the
                 input data
                - nx (int): is the number of input features to the neuron
                - m (int): is the number of examples
        Return:
            The private attributes __A1 and __A2, respectively
        """

        # Z1 = W1.X + b1
        Z1 = np.matmul(self.__W1, X) + self.__b1
        sigmoid_func = 1 / (1 + np.exp(-Z1))
        self.__A1 = sigmoid_func

        # In the case of the output neuron, the inputs are from layer A1
        # instead of X
        # Z2 = W2.__A1 + b2
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        sigmoid_func = 1 / (1 + np.exp(-Z2))
        self.__A2 = sigmoid_func

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Args:
            - Y: is a numpy.ndarray with shape (1, m) that contains the
              correct labels for the input data
            - A: is a numpy.ndarray with shape (1, m) containing the activated
              output of the neuron for each example
        Return:
            The cost of the model
        """
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        Args:
            - X: is a numpy.ndarray with shape (nx, m) that contains the input
              data
                - nx: is the number of input features to the neuron
                - m: is the number of examples
            - Y: is a numpy.ndarray with shape (1, m) that contains the correct
              labels for the input data
        Return:
            The neuron’s prediction and the cost of the network, respectively
        """
        # Generate forward propagation
        self.forward_prop(X)

        # Calculate cost
        labels = np.where(self.__A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A2)

        return labels, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network and
        Updates the private attributes __W1, __b1, __W2, and __b2
        Args:
            - X: is a numpy.ndarray with shape (nx, m) that contains the input
              data
                - nx: is the number of input features to the neuron
                - m: is the number of examples
            - Y: is a numpy.ndarray with shape (1, m) that contains the correct
              labels for the input data
            - A1: is the output of the hidden layer
            - A2: is the predicted output
            - alpha: is the learning rate
        """
        m = A1.shape[1]
        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = np.matmul(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Update W1 and b1
        self.__W1 = self.__W1 - (alpha * dW1)
        self.__b1 = self.__b1 - (alpha * db1)
        # Update W2 and b2
        self.__W2 = self.__W2 - (alpha * dW2)
        self.__b2 = self.__b2 - (alpha * db2)
