#!/usr/bin/env python3
"""Script to slice a matrix along a specific axes"""


def np_slice(matrix, axes={}):
    """Returns a matrix sliced in specific axes"""
    new_matrix = []
    index = 0
    for key, value in axes.items():
        while index < key:
            new_matrix.append(slice(None))
            index += 1
        new_matrix.append(slice(*value))
        index += 1
    return matrix[tuple(new_matrix)]
