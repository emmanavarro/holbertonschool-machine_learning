#!/usr/bin/env python3
"""
Pooling on images
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images
    Args:
        - images is a numpy.ndarray with shape (m, h, w, c) containing multiple
          images
            :m: is the number of images
            :h: is the height in pixels of the images
            :w: is the width in pixels of the images
            :c: is the number of channels in the image
        - kernel_shape is a tuple of (kh, kw) containing the kernel shape for
          the pooling
            :kh: is the height of the kernel
            :kw: is the width of the kernel
        - stride is a tuple of (sh, sw)
            :sh: is the stride for the height of the image
            :sw: is the stride for the width of the image
        - mode indicates the type of pooling
            :max: indicates max pooling
            :avg: indicates average pooling
    Note: You are only allowed to use two for loops; any other loops of any
          kind are not allowed
    Returns: a numpy.ndarray containing the pooled images
    """
    m, h = images.shape[0], images.shape[1]
    w, c = images.shape[2], images.shape[3]
    kh, kw = kernel_shape[0], kernel_shape[1]
    sh, sw = stride[0], stride[1]

    # Output matrix height and width
    oh = int(((h - kh) / sh) + 1)
    ow = int(((w - kw) / sw) + 1)

    # Creating the output matrix with shape (m, oh, ow, c)
    output = np.zeros((m, oh, ow, c))

    # Loop over every pixel in the output
    for x in range(oh):
        for y in range(ow):
            x0 = x * sh
            y0 = y * sw
            x1 = x0 + kh
            y1 = y0 + kw
            max_pool = np.max
            if mode == 'avg':
                max_pool = np.mean
            output[:, x, y] = max_pool(images[:, x0:x1, y0:y1],
                                       axis=(1, 2))

    return output
