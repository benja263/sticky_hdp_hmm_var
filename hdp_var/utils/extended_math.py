"""
Module containing functions for numerically stable calculations
As follows from: Numerically Stable Hidden Markov Model Implementation Tobias Mann 2006
"""
from numpy import exp, full, log, nan
from numpy import isnan, zeros, array, add


def elog(x):
    """

    :param x:
    :return:
    """
    if x == 0.0:
        return nan
    return log(x)


def eexp(x):
    """

    :param x:
    :return:
    """
    if isnan(x):
        return 0.0
    return exp(x)


def elnsum(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    if isnan(x) or isnan(y):
        if isnan(x):
            return y
        return x
    if x > y:
        return x + elog(1 + exp(y - x))
    return y + elog(1 + exp(x - y))


def elogproduct(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    if isnan(x) or isnan(y):
        return nan
    return x + y


def ex_log(x):
    """

    :param array x:
    :return:
    """
    res = full(fill_value=nan, shape=x.shape)
    log(x, where=x > 0.0, out=res, dtype=array)
    return res


def ex_exp(x):
    """

    :param array x:
    :return:
    """
    res = zeros(x.shape)
    exp(x, where=~isnan(x), out=res)
    return res


def ex_log_product(elog_x, elog_y):
    """

    :param array elog_x: output of extended_log(x)
    :param array elog_y:output of extended_log(y)
    :return:
    """
    res = full(fill_value=nan, shape=elog_x.shape)
    add(elog_x, elog_y, where=(~isnan(elog_x)) & (~isnan(elog_y)), out=res)
    return res


