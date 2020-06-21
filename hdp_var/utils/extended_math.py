"""
Module containing functions for numerically stable calculations
As follows from: Numerically Stable Hidden Markov Model Implementation Tobias Mann 2006
"""
import numpy as np


def elog(x):
    if x == 0.0:
        return np.nan
    return np.log(x)


def eexp(x):
    if np.isnan(x):
        return 0.0
    return np.exp(x)


def elnsum(x, y):
    if np.isnan(x) or np.isnan(y):
        if np.isnan(x):
            return y
        return x
    if x > y:
        return x + elog(1 + np.exp(y - x))
    return y + elog(1 + np.exp(x - y))


def elogproduct(x, y):
    if np.isnan(x) or np.isnan(y):
        return np.nan
    return x + y


def ex_log(x):
    """

    :param np.array x:
    :return:
    """
    res = np.full(fill_value=np.nan, shape=x.shape)
    np.log(x, where=x > 0.0, out=res)
    return res


def ex_exp(x):
    """

    :param np.array x:
    :return:
    """
    res = np.zeros(x.shape)
    np.exp(x, where=~np.isnan(x), out=res)
    return res


def ex_log_product(elog_x, elog_y):
    """

    :param np.array elog_x: output of extended_log(x)
    :param np.array elog_y:output of extended_log(y)
    :return:
    """
    res = np.full(fill_value=np.nan, shape=elog_x.shape)
    np.add(elog_x, elog_y, where=(~np.isnan(elog_x)) & (~np.isnan(elog_y)), out=res)
    return res


