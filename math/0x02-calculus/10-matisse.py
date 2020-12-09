#!/usr/bin/env python3
"""Function that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """
    Calculates the derivative of a polynomial
    """
    if poly == [] or type(poly) is not list or len(poly) is 0:
        return None

    for term in poly:
        if not isinstance(term, (int, float)):
            return None

    if len(poly) is 1:
        return [0]

    deriv_pol = []
    for x in range(len(poly)):
        deriv_pol.append(poly[x] * x)
    return deriv_pol[1:]
