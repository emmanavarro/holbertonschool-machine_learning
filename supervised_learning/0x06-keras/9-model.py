#!/usr/bin/env python3
"""
Save and load a model
"""

import tensorflow.keras as keras


def save_model(network, filename):
    """
    Saves an entire model
    Args:
        - network: is the model to save
        - filename: is the path of the file that the model should be saved to
    Returns: None
    """
    network.save(filename)
    return None

def load_model(filename):
    """
    Loads an entire model
    Args:
        - filename: is the path of the file that the model should be loaded
          from
    Returns: the loaded model
    """
    network = keras.models.load_model(filename)
    return network
