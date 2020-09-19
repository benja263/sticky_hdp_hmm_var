"""
Module containing helper functions
"""
import numpy as np


def normalize_vec(x):
    """
    Return normalized vector and normalization constant
    :param np.array x:
    :return:
    """
    norm_const = np.sum(x)
    return x / norm_const, norm_const
