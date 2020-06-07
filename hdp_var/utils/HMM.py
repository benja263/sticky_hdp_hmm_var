"""
Module containing HMM related functions
"""
import numpy as np


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
    log_likelihood = np.zeros((L, T))
    for k in range(L):
        chol_sigma = np.linalg.cholesky(theta['Sigma'][:, :, k])
        mu = chol_sigma.dot(data['Y'] - theta['A'][:, :, k].dot(data['X']))
        log_likelihood[k] = -0.5 * np.sum(mu ** 2, axis=0) + np.sum(np.log(np.diag(chol_sigma)))
    normalizer = np.max(log_likelihood, axis=0)
    log_likelihood -= normalizer

    likelihoods = np.exp(log_likelihood)
    normalizer -= (D / 2) * np.log(2 * np.pi)
    # forward pass to integrate over the state sequence and the log probability of the evidence
    total_log_likelihood = forwards_messaging((L, T), likelihoods, pi_0, pi_z, normalizer)
    return likelihoods, total_log_likelihood


def forwards_messaging(size, likelihoods, pi_0, pi_z, normalizer=None):
    """
    Run a forward pass and return probability of the evidence
    :param size:
    :param likelihoods:
    :param pi_0:
    :param pi_z:
    :param normalizer:
    :return:
    """
    L, T = size
    fwd_msg = np.zeros((L, T))
    fwd_msg[:, 0] = np.multiply(likelihoods[:, 1], pi_0)
    # normalize
    sum_fwd_msg = np.sum(fwd_msg)
    fwd_msg[:, 0] /= sum_fwd_msg

    if normalizer is None:
        normalizer = np.zeros(T)
    # add the constant for normalizing the forward message
    # murphy's book page 611 Z is the constant
    normalizer[0] += np.log(sum_fwd_msg)
    # compute messages forward in time
    for t in range(T - 1):
        # integrate out z[t]
        partial_marg_likelihood = pi_z.transpose().dot(fwd_msg[:, t])
        # multiply likelihood by incoming message
        fwd_msg[:, t + 1] = np.multiply(partial_marg_likelihood, likelihoods[:, t + 1])
        sum_fwd_msg = np.sum(fwd_msg[:, t + 1])
        fwd_msg[:, t + 1] /= sum_fwd_msg
        normalizer[t + 1] += np.log(sum_fwd_msg)
    # return total log-likelihood
    return np.sum(normalizer)


def backwards_messaging(size, pi_z, likelihoods):
    """
    Run a backwards pass and return the partial marginal likelihoods
    :param size:
    :param pi_z:
    :param likelihoods:
    :return:
    """
    L, T = size
    back_msg = np.zeros((L, T))
    back_msg[:, -1] = np.ones(L)

    partial_marg_likelihoods = np.zeros(back_msg.shape)

    for t in range(T - 2, -1, -1):
        # multiply likelihood by incoming message
        partial_marg_likelihoods[:, t + 1] = np.multiply(likelihoods[:, t + 1], back_msg[:, t + 1])
        # integrate over z(t)
        back_msg[:, t] = pi_z.dot(partial_marg_likelihoods[:, t + 1])
        # normalize
        back_msg[:, t] /= np.sum(back_msg[:, t])
    # for t=0
    partial_marg_likelihoods[:, 0] = np.multiply(likelihoods[:, 0], back_msg[:, 0])
    return partial_marg_likelihoods


def viterbi(L, data, theta,pi_0, pi_z):
    """

    :return:
    """
    likelihoods, _ = compute_likelihoods(L, data, theta, pi_0, pi_z)
    T = likelihoods.shape[1]
    delta, psi = np.zeros(likelihoods.shape), np.zeros(likelihoods.shape, dtype=int)
    state_sequence = np.zeros(T, dtype=int)
    normalizer = np.ones(T)

    delta[:, 0] = np.multiply(pi_0, likelihoods[:, 0])
    normalizer[0] = np.sum(delta[:, 0])
    delta[:, 0] /= normalizer[0]

    for t in range(1, T):
        for j in range(L):
            # todo invented partial marg name - need to see what it means
            partial_marg = np.multiply(delta[:, t - 1], pi_z[:, j])
            delta[j, t], psi[j, t] = np.max(partial_marg) * likelihoods[j, t], np.argmax(partial_marg)
        normalizer[t] = np.sum(delta[:, t])
        delta[:, t] /= normalizer[t]

    state_sequence[T-1] = np.argmax(delta[:, T-1])
    for t in range(T - 2, -1, -1):
        state_sequence[t] = psi[state_sequence[t+1], t+1]
    return state_sequence
