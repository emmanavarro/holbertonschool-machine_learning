#!/usr/bin/env python3
"""
Pooling Forward Prop
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network
    Args:
        - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
          containing the output of the previous layer
            m: is the number of examples
            h_prev: is the height of the previous layer
            w_prev: is the width of the previous layer
            c_prev: is the number of channels in the previous layer
        - kernel_shape is a tuple of (kh, kw) containing the size of the kernel
          for the pooling
            :kh: is the kernel height
            :kw: is the kernel width
        - stride is a tuple of (sh, sw) containing the strides for the pooling
            :sh: is the stride for the height
            :sw: is the stride for the width
        - mode is a string containing either max or avg, indicating whether to
          perform maximum or average pooling, respectively
    Returns: the output of the pooling layer
    """
    # number of images
    n_images = A_prev.shape[0]

    # input_width and input_height
    i_h = A_prev.shape[1]
    i_w = A_prev.shape[2]

    # images channel
    i_c = A_prev.shape[3]

    # kernel_width and kernel_height
    k_h = kernel_shape[0]
    k_w = kernel_shape[1]

    # stride_height and stride_width
    s_h = stride[0]
    s_w = stride[1]

    # output_height and output_width
    o_h = int((i_h - k_h) / s_h) + 1
    o_w = int((i_w - k_w) / s_w) + 1

    # creating outputs of size: [n_images,  o_h  ⊛  o_w  ⊛  k_c ⊛  i_c]
    outputs = np.zeros((n_images, o_h, o_w, i_c))

    # vectorizing the n_images into an array (creating a new dimension)
    imgs_arr = np.arange(0, n_images)

    # funtion selector
    funct = np.max
    if (mode == "avg"):
        funct = np.average

    # iterating over the output array and generating the pooling
    for x in range(o_h):
        for y in range(o_w):
            x0 = x * s_h
            y0 = y * s_w
            x1 = x0 + k_h
            y1 = y0 + k_w
            outputs[imgs_arr, x, y] = funct(A_prev[imgs_arr, x0: x1, y0: y1],
                                            axis=(1, 2))

    return outputs
