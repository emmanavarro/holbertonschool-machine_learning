#!/usr/bin/env python3
"""
Valid convolution
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images
    Args:
        - images is a numpy.ndarray with shape (m, h, w) containing multiple
          grayscale images:
            :m: is the number of images
            :h: is the height in pixels of the images
            :w: is the width in pixels of the images
        - kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
          for the convolution
            :kh: is the height of the kernel
            :kw: is the width of the kernel
    Note: You are only allowed to use two for loops; any other loops of any
          kind are not allowed
    Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    # Output matrix height and width
    oh = h - kh + 1
    ow = w - kw + 1

    # Creating the output matrix with shape (oh, ow)
    output = np.zeros((m, oh, ow))

    # Loop over every pixel in the output
    for x in range(ow):
        for y in range(oh):
            x1 = x + kh
            y1 = y + kw
            output[:, x, y] = np.sum(images[:, x:x1, y:y1] * kernel,
                                     axis=(1, 2))

    return output
