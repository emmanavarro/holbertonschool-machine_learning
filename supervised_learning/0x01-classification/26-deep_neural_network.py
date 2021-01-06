#!/usr/bin/env python3
"""
Deep neural network
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification
    """
    def __init__(self, nx, layers):
        """
        Class constructor
        Args:
            - nx: is the number of input features
            - layers: is a list representing the number of nodes in each layer
              of the network
        Public instance attributes:
            - L: The number of layers in the neural network.
            - cache: A dictionary to hold all intermediary values of the
              network.
            - weights: A dictionary to hold all weights and biased of the
              network.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for lay in range(self.__L):
            if not isinstance(layers[lay], int) or layers[lay] <= 0:
                raise TypeError("layers must be a list of positive integers")
            if lay == 0:
                # He et al. initialization for weights in first layer
                He = (np.random.randn(layers[lay], nx)
                      * np.sqrt(2 / nx))
                self.__weights["W{}".format(lay + 1)] = He
            else:
                # He et al. initialization for weights
                He = (np.random.randn(layers[lay], layers[lay - 1])
                      * np.sqrt(2 / layers[lay - 1]))
                self.__weights["W{}".format(lay + 1)] = He
            # Zero initialization for biases
            self.__weights["b{}".format(lay + 1)] = np.zeros((layers[lay], 1))

    @property
    def L(self):
        """ Getter method for __L """
        return self.__L

    @property
    def cache(self):
        """ Getter method for __cache """
        return self.__cache

    @property
    def weights(self):
        """ Getter method for __weights """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network, updates the
        private attribute __cache.
        - X: is a numpy.ndarray with shape (nx, m) that contains the input data
            - nx: is the number of input features to the neuron
            - m: is the number of examples
        Return:
            The output of the neural network and the cache, respectively
        """
        self.__cache["A0"] = X
        for layer in range(self.__L):
            weights = self.__weights["W{}".format(layer + 1)]
            a_ = self.__cache["A{}".format(layer)]
            biases = self.__weights["b{}".format(layer + 1)]

            # Forward propagation function
            Z = np.matmul(weights, a_) + biases

            # Sigmoid activation function
            sigmoid = 1 / (1 + np.exp(-Z))

            # Update cache
            self.__cache["A{}".format(layer + 1)] = sigmoid

        return self.__cache["A{}".format(layer + 1)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.
        Args:
            - Y: is a numpy.ndarray with shape (1, m) that contains the correct
                 labels for the input data.
            - A: is a numpy.ndarray with shape (1, m) containing the activated
                 output of the neuron for each example.
        To avoid division by zero errors, it is used 1.0000001 - A instead of
        1 - A.
        Return:
            The cost of the model.
        """
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions.
        Args:
            - X: is a numpy.ndarray with shape (nx, m) that contains the input
                 data.
                - nx: is the number of input features to the neuron.
                - m: is the number of examples.
            - Y: is a numpy.ndarray with shape (1, m) that contains the correct
                 labels for the input data.
        Return:
            The neuron’s prediction and the cost of the network, respectively.
        """
        # Generate forward propagation
        self.forward_prop(X)
        # Calculate cost
        cost = self.cost(Y, self.__cache["A{}".format(self.__L)])
        # Evaluate
        labels = np.where(self.__cache["A{}".format(self.__L)] >= 0.5, 1, 0)

        return labels, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network, updates
        the private attribute __weights.
        Args:
            - Y: is a numpy.ndarray with shape (1, m) that contains the correct
                 labels for the input data.
            - cache: is a dictionary containing all the intermediary values of
                     the network.
            - alpha: is the learning rate.
        """
        m = Y.shape[1]
        dZ = self.__cache["A{}".format(self.__L)] - Y

        for layer in range(self.__L, 0, -1):
            weights = self.__weights["W{}".format(layer)]
            A = self.__cache["A{}".format(layer - 1)]
            biases = self.__weights["b{}".format(layer)]

            dW = np.matmul(dZ, A.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dZ = np.matmul(weights.T, dZ) * A * (1 - A)

            # Update __weights
            self.__weights["W{}".format(layer)] = weights - (alpha * dW)
            self.__weights["b{}".format(layer)] = biases - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network by updating the private attributes
        __weights and __cache.
        Args:
            - X: is a numpy.ndarray with shape (nx, m) that contains the input
                 data.
                - nx: is the number of input features to the neuron.
                - m: is the number of examples.
            - Y: is a numpy.ndarray with shape (1, m) that contains the correct
                 labels for the input data.
            - iterations: is the number of iterations to train over.
            - alpha: is the learning rate.
            - verbose: is a boolean that defines whether or not to print
              information about the training.
            - graph: is a boolean that defines whether or not to graph
              information about the training once the training has completed.
        Return:
            The evaluation of the training data after iterations of training
            have occurred.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        cost_list = []
        step_list = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            cost = self.cost(Y, self.__cache["A{}".format(self.__L)])
            if verbose is True:
                if (i % step == 0 or step == iterations):
                    cost_list.append(cost)
                    step_list.append(i)
                    print("Cost after {} iterations: {}".format(i, cost))

        if graph is True:
            x = step_list
            y = cost_list
            plt.plot(x, y, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format.
        Arg:
            - filename: is the file to which the object should be saved.
        """
        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object.
        Arg:
            - filename: is the file from which the object should be loaded.
        Return:
            The loaded object, or None if filename doesn’t exist.
        """
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
