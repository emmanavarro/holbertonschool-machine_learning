#!/usr/bin/env python3
"""Script to concatenate two arrays"""


def cat_arrays(arr1, arr2):
    """Concatenates two arrays"""
    concat_array = []
    concat_array.extend(arr1 + arr2)
    return concat_array
