#!/usr/bin/env python3
""" Creates a class Poisson that represents a poisson distribution """


class Poisson:
    """ Class constructor """
    def __init__(self, data=None, lambtha=1.):
        """ Sets the instance attribute lambtha """
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pmf(self, k):
        """ Calculates the value of the PMF for a
            given number of “successes” """
        e = 2.7182818285

        k = int(k)
        if k < 0:
            return 0

        factorial = 1
        for n in range(1, k + 1):
            factorial *= n

        pmf_res = ((self.lambtha ** k) * (e ** (-self.lambtha))) / factorial
        return pmf_res

    def cdf(self, k):
        """ Calculates the value of the CDF for a
            given number of “successes” """
        e = 2.7182818285

        k = int(k)
        if k < 0:
            return 0

        summation = 0
        for n in range(k + 1):
            factorial = 1
            for m in range(1, n + 1):
                factorial *= m
            summation += (self.lambtha ** n) / factorial

        cdf_res = (e ** (-self.lambtha)) * summation
        return cdf_res
