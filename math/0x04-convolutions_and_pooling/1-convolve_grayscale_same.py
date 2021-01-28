#!/usr/bin/env python3
"""
Same convolution
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
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
    Notes: if necessary, the image should be padded with 0’s
           You are only allowed to use two for loops; any other loops of any
           kind are not allowed
    Returns: a numpy.ndarray containing the convolved images
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    # Calculate the right padding for the output
    ph = max(int((kh - 1) / 2), int(kh / 2))
    pw = max(int((kw - 1) / 2), int(kw / 2))

    # Creating the output matrix with shape (m, h, w) as the inital input
    output = np.zeros((m, h, w))

    # Creating the pad of zeros around the output matrix
    pad_img = np.pad(images,
                     pad_width=((0, 0), (ph, ph), (pw, pw)),
                     mode='constant',
                     constant_values=0)

    # Loop over every pixel in the output
    for x in range(w):
        for y in range(h):
            x1 = x + kh
            y1 = y + kw
            output[:, y, x] = np.sum(pad_img[:, y:y1, x:x1] * kernel,
                                     axis=(1, 2))

    return output
