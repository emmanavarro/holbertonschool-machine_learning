#!/usr/bin/env python3
"""Script to concatenate two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Returns two concatenated matrices"""

    mat1_copy = [row[:] for row in mat1]
    mat2_copy = [row[:] for row in mat2]

    if (axis == 0) and (len(mat1[0]) == len(mat2[0])):
        return mat1_copy + mat2_copy
    elif (axis == 1) and (len(mat1) == len(mat2)):
        matrix_concat = []
        for i in range(len(mat1)):
            matrix_concat.append(mat1_copy[i] + mat2_copy[i])
        return matrix_concat
    else:
        return None
