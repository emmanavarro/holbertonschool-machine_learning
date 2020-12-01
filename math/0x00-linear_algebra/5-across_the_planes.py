#!/usr/bin/env python3
"""Script to add two matrices 2D element-wise"""


def add_matrices2D(mat1, mat2):
    """Adds two matrices element-wise"""
    if (len(mat1) == len(mat2)) and (len(mat1[0]) == len(mat2[0])):
        matrix_sum = []
        for i in range(len(mat1)):
            rows = []
            for j in range(len(mat1[0])):
                rows.append(mat1[i][j] + mat2[i][j])
            matrix_sum.append(rows)
        return matrix_sum
    else:
        return None
