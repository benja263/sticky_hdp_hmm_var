"""
Module containing HMM related functions
"""
from scipy.linalg import cholesky
import numpy as np
from utils.math.c_extensions import e_array_log, forwards_pass, backwards_pass, load_c_lib, e_array_exp


def compute_likelihoods(L, data, theta, pi_0, pi_z, c_lib=None):
    """
    Compute log-likelihoods for each z[t] where z[t] equals the HMM state at time t.
    Log-likelihood is calculated using a MNIW prior
    """
    if c_lib is None:
        c_lib = load_c_lib()
    D, T = np.shape(data['Y'])
    log_likelihoods = np.zeros((L, T))
    for k in range(L):
        inv_sigma = cholesky(theta['inv_sigma'][:, :, k])
        mu = inv_sigma @ (data['Y'] - theta['A'][:, :, k] @ data['X'])
        log_likelihoods[k, :] = -0.5 * np.sum(mu ** 2, axis=0) + np.sum(np.log(np.diag(inv_sigma)))

    normalizer = np.max(log_likelihoods, axis=0)
    likelihoods = e_array_exp(log_likelihoods - normalizer, c_lib)
    normalized_log_likelihoods = log_likelihoods - normalizer
    normalizer -= (D / 2.0) * np.log(2.0 * np.pi)
    # forward pass to integrate over the state sequence and the log probability of the evidence
    _, sequence_log_likelihood = forwards_messaging((L, T), log_likelihoods, pi_0, pi_z, normalizer, c_lib)
    return likelihoods, normalized_log_likelihoods, sequence_log_likelihood


def forwards_messaging(size, log_likelihoods, pi_0, pi_z, normalizer=None, c_lib=None):
    """
    Run a forwards pass and return probability of the evidence
    :return:
    """
    if c_lib is None:
        c_lib = load_c_lib()
    L, T = size
    log_pi_z, log_pi_0 = e_array_log(pi_z, c_lib), e_array_log(pi_0, c_lib)
    log_alpha = np.zeros(size)
    log_alpha[:, 0] = np.add(log_pi_0, log_likelihoods[:, 0],
                             where=(~np.isnan(log_pi_0)) & (~np.isnan(log_likelihoods[:, 0])))

    log_alpha = forwards_pass(log_alpha, log_likelihoods, log_pi_z, c_lib)

    if normalizer is None:
        normalizer = np.zeros(T)
    normalizer += np.max(log_alpha, axis=0)
    return log_alpha, np.sum(normalizer)


def backwards_messaging(size, pi_z, log_likelihoods, c_lib=None):
    """
    Run a backwards pass and return the backwards messages
    """
    if c_lib is None:
        c_lib = load_c_lib()
    log_pi_z = e_array_log(pi_z, c_lib)
    log_messages = backwards_pass(np.zeros(size), log_likelihoods, log_pi_z, c_lib)
    return e_array_exp(log_messages - np.max(log_messages, axis=0), c_lib)


def viterbi(L, data, theta, pi_0, pi_z, c_lib):
    """
    Run the viterbi algorithm to get the most likely state-sequence
    :return:
    """
    likelihoods, _, _ = compute_likelihoods(L, data, theta, pi_0, pi_z)
    T = likelihoods.shape[1]
    delta, psi = np.zeros(likelihoods.shape), np.zeros(likelihoods.shape, dtype=int)
    state_sequence = np.zeros(T, dtype=int)
    normalizer = np.ones(T)

    delta[:, 0] = pi_0 * likelihoods[:, 0]
    normalizer[0] = np.sum(delta[:, 0])
    delta[:, 0] /= normalizer[0]

    for t in range(1, T):
        for j in range(L):
            temp = delta[:, t - 1] * pi_z[:, j]
            delta[j, t], psi[j, t] = np.max(temp) * likelihoods[j, t], np.argmax(temp)
        normalizer[t] = np.sum(delta[:, t])
        delta[:, t] /= normalizer[t]

    state_sequence[T - 1] = np.argmax(delta[:, T - 1])
    for t in range(T - 2, -1, -1):
        state_sequence[t] = psi[state_sequence[t + 1], t + 1]
    return state_sequence






