#!/usr/bin/env python3
"""Function that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial
    """
    if poly is None or type(poly) is not list or poly is []:
        return None

    deriv_pol = []
    for x in range(1, len(poly)):
        deriv_pol.append(poly[x] * x)
    return deriv_pol

    if deriv_pol == 0:
        return None
