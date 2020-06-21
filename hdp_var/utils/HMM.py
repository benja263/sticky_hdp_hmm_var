"""
Module containing HMM related functions
"""
from scipy.linalg import cholesky, inv

from hdp_var.utils.extended_math import *


def compute_likelihoods(L, data, theta, pi_0, pi_z):
    """
    Compute log-likelihoods for each z[t] where z[t] equals the HMM state at time t.
    Log-likelihood is calculated using a MNIW prior
    :param L:
    :param data:
    :param theta:
    :param pi_0:
    :param pi_z:
    :return:
    """
    D, T = np.shape(data['Y'])
    log_likelihoods = np.zeros((L, T))
    for k in range(L):
        inv_sigma = cholesky(theta['inv_sigma'][:, :, k])
        mu = inv_sigma @ (data['Y'] - theta['A'][:, :, k] @ data['X'])
        log_likelihoods[k, :] = -0.5 * np.sum(mu ** 2, axis=0) + np.sum(np.log(np.diag(inv_sigma)))

    normalizer = np.max(log_likelihoods, axis=0)
    # log_likelihoods -= normalizer
    normalizer -= (D / 2.0) * np.log(2.0 * np.pi)
    # forward pass to integrate over the state sequence and the log probability of the evidence
    forward_messages, sequence_log_likelihood = forwards_messaging((L, T), log_likelihoods, pi_0, pi_z, normalizer)
    return log_likelihoods, sequence_log_likelihood


def forwards_messaging(size, log_likelihoods, pi_0, pi_z, normalizer=None):
    """
    Run a forward pass and return probability of the evidence
    :param size:
    :param log_likelihoods:
    :param pi_0:
    :param pi_z:
    :param normalizer: Initialize normalization constant to be that due to the likelihood
    :return:
    """
    L, T = size
    # initializing with very small numbers to avoid taking the log of 0
    log_pi_z, log_pi_0 = ex_log(pi_z), ex_log(pi_0)

    log_alpha = np.zeros(size)

    log_alpha[:, 0] = ex_log_product(log_pi_0, log_likelihoods[:, 0])

    if normalizer is None:
        normalizer = np.zeros(T)
    # add the constant for normalizing the forward message
    # murphy's book page 611 Z is the constant

    for t in range(T - 1):
        for j in range(L):
            accumulator = np.nan
            for i in range(L):
                accumulator = elnsum(accumulator, elogproduct(log_alpha[i, t], log_pi_z[i, j]))
            log_alpha[j, t + 1] = elogproduct(accumulator, log_likelihoods[j, t])
    normalizer += np.max(log_alpha, axis=0)
    return log_alpha, np.sum(normalizer)


def backwards_messaging(size, pi_z, log_likelihoods):
    """
    Run a backwards pass and return the partial marginal likelihoods
    :param size:
    :param pi_z:
    :param log_likelihoods:
    :return:
    """
    L, T = size
    log_pi_z = ex_log(pi_z)
    ex_ln_beta = np.zeros(size)
    for t in reversed(range(T - 1)):
        # log_beta = np.full(fill_value=np.nan, shape=L)
        for i in range(L):
            log_beta = np.nan
            res = ex_log_product(log_pi_z[i, :], ex_log_product(log_likelihoods[:, t + 1], ex_ln_beta[:, t + 1]))
            for j in range(L):
                log_beta = elnsum(log_beta, res[j])
            ex_ln_beta[i, t] = log_beta
    return ex_ln_beta


def viterbi(L, data, theta, pi_0, pi_z):
    """

    :return:
    """
    log_likelihoods, _ = compute_likelihoods(L, data, theta, pi_0, pi_z)
    likelihoods = np.exp(log_likelihoods)
    T = likelihoods.shape[1]
    delta, psi = np.zeros(likelihoods.shape), np.zeros(likelihoods.shape, dtype=int)
    state_sequence = np.zeros(T, dtype=int)
    normalizer = np.ones(T)

    delta[:, 0] = pi_0 * likelihoods[:, 0]
    normalizer[0] = np.sum(delta[:, 0])
    delta[:, 0] /= normalizer[0]

    for t in range(1, T):
        for j in range(L):
            # todo invented partial marg name - need to see what it means
            partial_marg = delta[:, t - 1] * pi_z[:, j]
            delta[j, t], psi[j, t] = np.max(partial_marg) * likelihoods[j, t], np.argmax(partial_marg)
        normalizer[t] = np.sum(delta[:, t])
        delta[:, t] /= normalizer[t]

    state_sequence[T - 1] = np.argmax(delta[:, T - 1])
    for t in range(T - 2, -1, -1):
        state_sequence[t] = psi[state_sequence[t + 1], t + 1]
    return state_sequence
