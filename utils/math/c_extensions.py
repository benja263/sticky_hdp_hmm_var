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


def ensure_contiguous(array):
    """
    Ensure that array is contiguous
    :param array:
    :return:
    """
    return np.ascontiguousarray(array) if not array.flags['C_CONTIGUOUS'] else array


def np_to_double_ptr(np_array):
    """
    Convert numpy ndarray to C++ double pointer
    :param np.ndarray np_array:
    :return:
    """
    np_array = ensure_contiguous(np_array)
    dim1, dim2 = np_array.shape[0], np_array.shape[1]
    ptr = ctypes.POINTER(ctypes.c_double)
    double_ptr = (ptr * dim1)()
    for i in range(dim1):
        double_ptr[i] = (ctypes.c_double * dim2)()
        for j in range(dim2):
            double_ptr[i][j] = np_array[i, j]
    return double_ptr


def double_ptr_to_np(double_ptr, shape):
    """
    Converts C++ double pointer to numpy ndarray
    :param double_ptr:
    :param shape:
    :return:
    """
    dim1, dim2 = shape
    np_ndarray = np.zeros(shape, dtype=np.float64)
    for i in range(dim1):
        np_ndarray[i] = np.ctypeslib.as_array(double_ptr[i], shape=(dim2,))
    return np_ndarray


def e_array_log(array, c_lib):
    """
    Call 'ExtendedArrayLog' function in C++ from Python
    :param np.ndarray array:
    :param c_lib:
    :return:
    """
    double_ptr_array = np_to_double_ptr(array)
    _ = c_lib.ExtendedArrayLog(double_ptr_array, ctypes.c_uint(array.shape[0]), ctypes.c_uint(array.shape[1]))
    return double_ptr_to_np(double_ptr_array, array.shape)


def e_array_exp(array, c_lib):
    """
    Call 'ExtendedArrayExp' function in C++ from Python
    :param np.ndarray array:
    :param c_lib:
    :return:
    """
    double_ptr_array = np_to_double_ptr(array)
    _ = c_lib.ExtendedArrayExp(double_ptr_array, ctypes.c_uint(array.shape[0]), ctypes.c_uint(array.shape[1]))
    return double_ptr_to_np(double_ptr_array, array.shape)


def forwards_pass(log_alpha, log_obs, log_pi_z, c_lib):
    """
    Call 'Forwards' function in C++ from Python
    :param np.ndarray log_alpha:
    :param np.ndarray log_obs:
    :param np.ndarray log_pi_z:
    :param c_lib:
    :return:
    """
    log_alpha, log_obs, log_pi_z = ensure_contiguous(log_alpha), ensure_contiguous(log_obs), ensure_contiguous(log_pi_z)
    temp_alpha, temp_obs, temp_pi_z = np_to_double_ptr(log_alpha), np_to_double_ptr(log_obs), np_to_double_ptr(
        log_pi_z)
    _ = c_lib.forwards(temp_alpha, temp_obs, temp_pi_z, ctypes.c_uint(log_alpha.shape[0]),
                       ctypes.c_uint(log_alpha.shape[1]))
    return double_ptr_to_np(temp_alpha, log_alpha.shape)


def backwards_pass(log_beta, log_obs, log_pi_z, c_lib):
    """
    Call 'Backwards' function in C++ from Python
    :param np.ndarray log_beta:
    :param np.ndarray log_obs:
    :param np.ndarray log_pi_z:
    :param c_lib:
    :return:
    """
    log_beta, log_obs, log_pi_z = ensure_contiguous(log_beta), ensure_contiguous(log_obs), ensure_contiguous(log_pi_z)
    double_ptr_beta, double_ptr_obs, double_ptr_pi_z = np_to_double_ptr(log_beta), np_to_double_ptr(log_obs), np_to_double_ptr(log_pi_z)
    _ = c_lib.backwards(double_ptr_beta, double_ptr_obs, double_ptr_pi_z, ctypes.c_uint(log_beta.shape[0]),
                        ctypes.c_uint(log_beta.shape[1]))
    return double_ptr_to_np(double_ptr_beta, log_beta.shape)


def rand_gamma(alpha, c_lib):
    """
    Call 'rand_gamma' function from C++ in Python
    :param alpha:
    :param c_lib:
    :return:
    """
    alpha = ensure_contiguous(alpha)
    gamma_random_array = np.zeros(alpha.shape)
    _ = c_lib.rand_gamma(gamma_random_array.ctypes.data_as(ctypes.c_void_p), alpha.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_uint(len(alpha)))
    return np.ctypeslib.as_array(gamma_random_array, len(alpha))


def rand_dirichlet(alpha, c_lib):
    """
    Call 'rand_dirichlet' function from C++ in Python
    :param alpha:
    :param c_lib:
    :return:
    """
    alpha = ensure_contiguous(alpha)
    rand_dirichlet_array = np.zeros(alpha.shape)
    _ = c_lib.rand_dirichlet(rand_dirichlet_array.ctypes.data_as(ctypes.c_void_p), alpha.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_uint(len(alpha)))
    return np.ctypeslib.as_array(rand_dirichlet_array, len(alpha))
