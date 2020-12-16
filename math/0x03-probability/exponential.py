#!/usr/bin/env python3
""" Create a class Exponential that represents
    an exponential distribution """


class Exponential:
    """ Exponential class """

    def __init__(self, data=None, lambtha=1.):
        """ Class contructor """
        if data is not None:
            if not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            self.lambtha = 1 / mean
        else:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pdf(self, x):
        """ Calculates the value of the
            PDF for a given time period """
        e = 2.7182818285

        if x < 0:
            return 0

        pdf_val = self.lambtha * (e ** (-self.lambtha * x))
        return pdf_val

    def cdf(self, x):
        """ Calculates the value of the
            CDF for a given time period """
        e = 2.7182818285

        if x < 0:
            return 0

        cdf_val = 1 - e ** (-self.lambtha * x)
        return cdf_val
