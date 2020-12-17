#!/usr/bin/env python3
""" Creates a class Binomial that represents a binomial distribution """


class Binomial:
    """ Binomial class """
    def __init__(self, data=None, n=1, p=0.5):
        """ Class constructor """
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = 0
            for x in data:
                variance += (x - mean) ** 2
            variance = variance / len(data)
            p = 1 - (variance / mean)
            self.n = int(round(mean / p))
            self.p = float(mean / self.n)
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            self.n = int(n)
            if (p <= 0) or (p >= 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)
