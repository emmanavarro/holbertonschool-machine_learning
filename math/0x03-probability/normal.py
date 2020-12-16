#!/usr/bin/env python3
""" Create a class Normal that represents a normal distribution """


class Normal:
    """ Normal class """
    def __init__(self, data=None, mean=0., stddev=1.):
        """ Class constructor """
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            list_data = [(x - self.mean) ** 2 for x in data]
            self.stddev = (sum(list_data) / len(data)) ** 0.5
        else:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

    def z_score(self, x):
        """ Calculates the z-score of a given x-value """
        z_result = (x - self.mean) / self.stddev
        return z_result

    def x_value(self, z):
        """ Calculates the x-value of a given z-score """
        x_result = (self.stddev * z) + self.mean
        return x_result
