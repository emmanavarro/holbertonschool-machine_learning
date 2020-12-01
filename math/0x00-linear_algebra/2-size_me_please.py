#!/usr/bin/env python3
""" Script to find the shape of a matrix """


def matrix_shape(matrix):
    """ Calculates the shape of a matrix """
    shape = []
    if type(matrix) == list:
        shape.append(len(matrix))
        shape += matrix_shape(matrix[0])
    else:
        pass
    return shape
