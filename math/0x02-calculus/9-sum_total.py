#!/usr/bin/env python3
"""Script to calculate a summation"""


def summation_i_squared(n):
    """
    Function that calculates the summation of i squared,
    from i = 1 to i = n
    """
    if n != int and n <= 0:
        return None
    else:
        res = sum((map(lambda res: res ** 2, range(1, n+1))))
        return res
