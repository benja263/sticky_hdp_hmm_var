"""
Data preparation module for Vector Auto Regression of the type:
Y[t] = A[t-1]*Y[t-1] + A[t-2]*Y[t-2] + ... + A[t-r]*Y[t-r] + e[t], where e is normally distributed with 0 mean and
covariance Sigma
"""
import numpy as np


def generate_data_structure(Y, order):
    """
    Return a dictionary as a data structure containing the input observations (Y) in a format suited for
     training the HDP-VAR model
    :param np.array Y: DxT time-series observation matrix where D is the number of dimensions and T is the number of time points
    :param int order: VAR order 'r'
    :return:
    """
    X = create_X_matrix(Y, order)
    data = {'Y': Y[:, order:], 'X': X}
    return data


def create_X_matrix(Y, order):
    """
    Generate matrix X such that X[t] = Y[t-1:t-order]
    Every column in X corresponds to the previous r observations (r = order) and Y equals the current observation
    :param np.array Y: DxT time-series observation matrix where D is the number of dimensions and T is the number of time points
    :param int order:
    :return:
    """
    D, T = np.shape(Y)
    X = np.zeros((D*order, T))
    for lag in range(1, order+1):
        row_indices = np.arange(D * (lag - 1), D * (lag - 1) + D)
        X[row_indices] = np.concatenate((np.zeros((D, min([lag, T]))), Y[:, np.arange(T - min([lag, T]))]), axis=1)
    return X[:, order:]



