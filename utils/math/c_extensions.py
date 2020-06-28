"""
Module interfacing between the C++ math extension functions and python
"""
import ctypes
import ctypes.util
import sys
import os

import numpy as np


def load_c_lib():
    """
    Load C shared library
    :return:
    """
    c_lib_path = ctypes.util.find_library(f"{os.path.dirname(os.path.abspath(__file__))}/c_extensions")
    try:
        c_lib = ctypes.CDLL(c_lib_path)
    except OSError:
        print("Unable to load the requested C library")
        sys.exit()
    return c_lib


def ensure_contiguous(x):
    if not x.flags['C_CONTIGUOUS']:
        return np.ascontiguousarray(x)
    return x


def np_2_double_pointer(np_array):
    """
    Convert numpy ndarray to C++ double pointer
    :param np.ndarray np_array:
    :return:
    """
    dim1, dim2 = np_array.shape[0], np_array.shape[1]
    double_pointer = ctypes.POINTER(ctypes.c_double)
    ptr = (double_pointer * dim1)()
    for i in range(dim1):
        ptr[i] = (ctypes.c_double * dim2)()
        for j in range(dim2):
            ptr[i][j] = np_array[i, j]
    return ptr


def double_pointer_2_np(double_pointer, shape):
    """
    Converts C++ double pointer to numpy ndarray
    :param double_pointer:
    :param shape:
    :return:
    """
    dim1, dim2 = shape
    res = np.zeros(shape, dtype=np.float64)
    for i in range(dim1):
        res[i] = np.ctypeslib.as_array(double_pointer[i], shape=(dim2,))
    return res


def e_array_log(x, c_lib):
    x = ensure_contiguous(x)
    temp = np_2_double_pointer(x)
    _ = c_lib.ExtendedArrayLog(temp, ctypes.c_uint(x.shape[0]), ctypes.c_uint(x.shape[1]))
    return double_pointer_2_np(temp, x.shape)


def e_array_exp(x, c_lib):
    x = ensure_contiguous(x)
    temp = np_2_double_pointer(x)
    _ = c_lib.ExtendedArrayExp(temp, ctypes.c_uint(x.shape[0]), ctypes.c_uint(x.shape[1]))
    return double_pointer_2_np(temp, x.shape)


def forwards_pass(log_alpha, log_obs, log_pi_z, c_lib):
    log_alpha, log_obs, log_pi_z = ensure_contiguous(log_alpha), ensure_contiguous(log_obs), ensure_contiguous(log_pi_z)
    temp_alpha, temp_obs, temp_pi_z = np_2_double_pointer(log_alpha), np_2_double_pointer(log_obs), np_2_double_pointer(
        log_pi_z)
    _ = c_lib.forwards(temp_alpha, temp_obs, temp_pi_z, ctypes.c_uint(log_alpha.shape[0]),
                       ctypes.c_uint(log_alpha.shape[1]))
    return double_pointer_2_np(temp_alpha, log_alpha.shape)


def backwards_pass(log_beta, log_obs, log_pi_z, c_lib):
    log_beta, log_obs, log_pi_z = ensure_contiguous(log_beta), ensure_contiguous(log_obs), ensure_contiguous(log_pi_z)
    temp_beta, temp_obs, temp_pi_z = np_2_double_pointer(log_beta), np_2_double_pointer(log_obs), np_2_double_pointer(log_pi_z)
    _ = c_lib.backwards(temp_beta, temp_obs, temp_pi_z, ctypes.c_uint(log_beta.shape[0]),
                        ctypes.c_uint(log_beta.shape[1]))
    return double_pointer_2_np(temp_beta, log_beta.shape)


def rand_gamma(alpha, c_lib):
    alpha = ensure_contiguous(alpha)
    res = np.zeros(alpha.shape)
    _ = c_lib.rand_gamma(res.ctypes.data_as(ctypes.c_void_p), alpha.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_uint(len(alpha)))
    return np.ctypeslib.as_array(res, len(alpha))


def rand_dirichlet(alpha, c_lib):
    alpha = ensure_contiguous(alpha)
    res = np.zeros(alpha.shape)
    _ = c_lib.rand_dirichlet(res.ctypes.data_as(ctypes.c_void_p), alpha.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_uint(len(alpha)))
    return np.ctypeslib.as_array(res, len(alpha))
