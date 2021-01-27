#!/usr/bin/env python3
"""
Save and Load a model's configuration
"""

import tensorflow.keras as keras


def save_config(network, filename):
    """
    Saves a model’s configuration
    Args:
        - network is the model whose configuration should be saved
        - filename is the path of the file that the configuration should be
          saved to
    Returns: None
    """
    json_model = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_model)
    return None


def load_config(filename):
    """
    Loads a model with a specific configuration
    Args:
        - filename is the path of the file containing the model’s
          configuration in JSON format
    Returns: the loaded model
    """
    with open(filename, 'r') as f:
        network_str = f.read()
    loaded_model = keras.models.model_from_json(network_str)
    return loaded_model
