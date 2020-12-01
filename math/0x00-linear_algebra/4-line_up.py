#!/usr/bin/env python3
"""Script to add two arrays element-wise"""


def add_arrays(arr1, arr2):
    """Adds two arrays element-wise"""
    if len(arr1) != len(arr2):
        return None
    else:
        matrix_sum = []
        for element in range(len(arr1)):
            matrix_sum.append(arr1[element] + arr2[element])
        return matrix_sum
