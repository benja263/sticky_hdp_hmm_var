"""
Data preparation module for Vector Auto Regression of the type:
Y[t] = A[t-1]*Y[t-1] + A[t-2]*Y[t-2] + ... + A[t-r]*Y[t-r] + e[t], where e is normally distributed with 0 mean and
covariance Sigma
"""
import numpy as np


def make_design_matrix(Y, order):
    """
    Generate matrix X such that X[t] = Y[t:t-order]
    Every column in X corresponds to the previous r observations (r = order) and Y equals the current observation
    :param np.array Y: DxT time-series observation matrix where D is the number of dimensions and T is the number of time points
    :param int order:
    :return:
    """
    D, T = np.shape(Y)
    X = np.zeros((D*order, T))
    for lag in range(1, order+1):
        ii = D*(lag-1)
        ind = np.arange(ii, ii+D)
        X[ind] = np.concatenate((np.zeros((D, np.min([lag, T]))), Y[:, np.arange(T - np.min([lag, T]))]), axis=1)
    valid_indices = np.ones(T, dtype=bool)
    valid_indices[:order] = 0
    return X, valid_indices


def generate_data_structure(Y, order):
    """
    Return a dictionary as a data structure containing the input observations (Y) in a format suited for
     training the HDP-VAR model
    :param np.array Y: DxT time-series observation matrix where D is the number of dimensions and T is the number of time points
    :param int order:
    :return:
    """
    X, valid = make_design_matrix(Y, order)
    data = {'Y': Y[:, valid], 'X': X[:, valid]}
    data['block_size'] = np.ones(data['Y'].shape[1], dtype=int)
    data['block_end'] = np.cumsum(data['block_size'])
    return data


# def prepare_mock_data(T, D, L, order):
#     """
#
#     :param D:
#     :param L:
#     :param order:
#     :return:
#     """
#     Y = np.zeros((D, T))
#     Y[:, :order] = np.ones((D, order))
#     A = np.zeros((D, D*order, L))
#     n = int(T / L)
#     state_sequence = np.zeros(T)
#     z = 0
#     for i in range(L):
#         temp = (i+1)*np.eye(D)
#         for r in range(1, order):
#             temp = np.concatenate((temp, ((i+1)/10)*np.eye(D)), axis=1)
#         if i % 2 == 0:
#             temp = -temp
#         A[:, :, i] = temp
#     for t in range(order, T - 1):
#         X_pred = Y[:, t - 1]
#         for r in range(2, order + 1):
#             X_pred = np.concatenate((X_pred, Y[:, t - r]), axis=0)
#         Y[:, t] = A[:, :, z].dot(X_pred)
#         state_sequence[t] = z
#         if t % n == 0:
#             z += 1
#     return Y, state_sequence, A

