#!/usr/bin/env python3
"""Function that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial
    """
    if type(poly) != list or len(poly) == 0:
        return None

    deriv_pol = []
    for x in range(len(poly)):
        deriv_pol.append(poly[x] * x)
    return deriv_pol[1:]

    if sum(deriv_pol) == 0:
        return [0]
