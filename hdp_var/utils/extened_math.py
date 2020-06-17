"""
Module containing functions for numerically stable calculations
As follows from: Numerically Stable Hidden Markov ModelImplementation Tobias Mann 2006
"""
import numpy as np


def extended_log(x):
    """

    :param np.array x:
    :return:
    """
    res = np.full(fill_value=np.nan, shape=x.shape)
    np.log(x, where=x > 0.0, out=res)
    return res


def extended_exp(x):
    """

    :param np.array x:
    :return:
    """
    res = np.zeros(x.shape)
    np.exp(x, where=~np.isnan(x), out=res)
    return res


def extended_logsum(x, y):
    """

    :param np.array x:
    :param np.array y:
    :return:
    """
    res = np.full(fill_value=np.nan, shape=x.shape)
    elog_x, elog_y = extended_log(x), extended_log(y)
    if np.isnan(elog_x).any() or np.isnan(elog_y).any():
        res[np.isnan(elog_x)] = elog_y[np.isnan(elog_x)]
        res[np.isnan(elog_y)] = elog_x[np.isnan(elog_y)]
        return res
    x_over_y, y_over_x = elog_x > elog_y, elog_y > elog_x
    if True in x_over_y:
        res[x_over_y] = elog_x[x_over_y] + extended_log(1.0 + np.exp(elog_y[x_over_y] - elog_x[x_over_y]))
    if True in y_over_x:
        res[y_over_x] = elog_y[y_over_x] + extended_log(1.0 + np.exp(elog_x[y_over_x] - elog_y[y_over_x]))
    return res



