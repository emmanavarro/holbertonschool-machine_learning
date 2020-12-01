#!/usr/bin/env python3
""" Script that returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """ Returns the transpose of a 2D matrix """
    rows = len(matrix)
    columns = len(matrix[0])
    transpose = []
    for column in range(columns):
        new_matrix = []
        for row in range(rows):
            new_matrix.append(matrix[row][column])
        transpose.append(new_matrix)
    return transpose
