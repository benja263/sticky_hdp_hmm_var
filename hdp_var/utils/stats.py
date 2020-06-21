"""

"""
import numpy as np
# from scipy.stats import wishart
# import numpy as np
#
# def inv_wishart(scale, df):
#     """
#
#     :param scale:
#     :param df:
#     :return:
#     """
#     if df >= scale.shape[1]:
#         chol_factor = np.linalg.lstsq(np.linalg.cholesky(scale).transpose(), np.eye(scale.shape))
#         W = wishart()


def right_divison(A, B):
    """

    :param A:
    :param B:
    :return:
    """
    return np.linalg.lstsq(B.transpose(), A.transpose(), rcond=None)[0].transpose()


def median_r_2(y, pred_y):
    """

    :param y:
    :param pred_y:
    :return:
    """
    MSE = ((y - pred_y)**2).sum(axis=1)
    mean_y = np.mean(y, axis=1)[:, np.newaxis]
    return np.median(1 - MSE / ((y - mean_y)**2).sum(axis=1))


