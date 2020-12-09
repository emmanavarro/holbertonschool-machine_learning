#!/usr/bin/env python3
"""Function that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial
    """
    if type(poly) != list or len(poly) == 0 or C != 0:
        return None

    integral = [C]
    for x in range(len(poly)):
        res = poly[x] / (x + 1)
        if res.is_integer():
            res = int(res)
        integral.append(res)
    return integral
