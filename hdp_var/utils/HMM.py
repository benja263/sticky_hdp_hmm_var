"""
Module containing HMM related functions
"""
import numpy as np
from scipy.linalg import cholesky, inv
from hdp_var.utils.extended_math import *
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


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
        inv_sigma = cholesky(inv(theta['sigma'][:, :, k]))
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
        log_alpha[:, t]
        log_alpha[:, t + 1] = log_sum_exp(log_pi_z.T + log_alpha[:, t] + log_likelihoods[:, t + 1])
        alpha[:, t + 1], Z[t] = log_softmax(log_alpha[:, t + 1])
    return alpha, np.sum(normalizer + Z)


def backwards_messaging(size, pi_z, log_likelihoods):
    """
    Run a backwards pass and return the partial marginal likelihoods
    :param size:
    :param pi_z:
    :param log_likelihoods:
    :return:
    """
    L, T = size
    # log_pi_z = np.full(fill_value=np.finfo(float).min, shape=pi_z.shape)
    # np.log(pi_z, where=pi_z > 0.0, out=log_pi_z)
    log_pi_z = ex_log(pi_z)
    log_beta = np.zeros(size)
    ex_ln_beta = np.zeros(size)
    beta = np.zeros(size)
    for t in reversed(range(T - 1)):
        # log_beta = np.full(fill_value=np.nan, shape=L)
        for i in range(L):
            log_beta = np.nan
            res = ex_log_product(log_pi_z[i, :], ex_log_product(log_likelihoods[:, t + 1], ex_ln_beta[:, t + 1]))
            for j in range(L):
                log_beta = elnsum(log_beta, res[j])
            ex_ln_beta[i, t] = log_beta
    # ex_ln_beta2 = np.zeros(size)
    # for t in reversed(range(T - 1)):
    #     partial_likelihood = ex_log_product(log_likelihoods[:, t + 1], ex_ln_beta2[:, t + 1])
    #     for i in range(L):
    #         log_beta = np.full(fill_value=np.nan, shape=L)
    #         res = ex_log_product(log_pi_z[i, :], partial_likelihood)
    #         for j in range(L):
    #             log_beta = ex_log_sum(log_beta, res)
    #     ex_ln_beta2[i, t] = log_beta
        # log_beta[:, t] = log_sum_exp_1d(log_beta[:, t+1] + log_likelihoods[:, t+1] + log_pi_z)
        # beta[:, t] = log_softmax(log_beta[:, t])[0]
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


def viterbi_path(prior, transmat, obslik, scaled=True, ret_loglik=False):
    '''Finds the most-probable (Viterbi) path through the HMM state trellis
    Notation:
        Z[t] := Observation at time t
        Q[t] := Hidden state at time t
    Inputs:
        prior: np.array(num_hid)
            prior[i] := Pr(Q[0] == i)
        transmat: np.ndarray((num_hid,num_hid))
            transmat[i,j] := Pr(Q[t+1] == j | Q[t] == i)
        obslik: np.ndarray((num_hid,num_obs))
            obslik[i,t] := Pr(Z[t] | Q[t] == i)
        scaled: bool
            whether or not to normalize the probability trellis along the way
            doing so prevents underflow by repeated multiplications of probabilities
        ret_loglik: bool
            whether or not to return the log-likelihood of the best path
    Outputs:
        path: np.array(num_obs)
            path[t] := Q[t]
    '''
    num_hid = obslik.shape[0]  # number of hidden states
    num_obs = obslik.shape[1]  # number of observations (not observation *states*)

    # trellis_prob[i,t] := Pr((best sequence of length t-1 goes to state i), Z[1:(t+1)])
    trellis_prob = np.zeros((num_hid, num_obs))
    # trellis_state[i,t] := best predecessor state given that we ended up in state i at t
    trellis_state = np.zeros((num_hid, num_obs), dtype=int)  # int because its elements will be used as indices
    path = np.zeros(num_obs, dtype=int)  # int because its elements will be used as indices

    trellis_prob[:, 0] = prior * obslik[:, 0]  # element-wise mult
    if scaled:
        scale = np.ones(num_obs)  # only instantiated if necessary to save memory
        scale[0] = 1.0 / np.sum(trellis_prob[:, 0])
        trellis_prob[:, 0] *= scale[0]

    trellis_state[:, 0] = 0  # arbitrary value since t == 0 has no predecessor
    for t in range(1, num_obs):
        for j in range(num_hid):
            trans_probs = trellis_prob[:, t - 1] * transmat[:, j]  # element-wise mult
            trellis_state[j, t] = trans_probs.argmax()
            trellis_prob[j, t] = trans_probs[trellis_state[j, t]]  # max of trans_probs
            trellis_prob[j, t] *= obslik[j, t]
        if scaled:
            scale[t] = 1.0 / np.sum(trellis_prob[:, t])
            trellis_prob[:, t] *= scale[t]

    path[-1] = trellis_prob[:, -1].argmax()
    for t in range(num_obs - 2, -1, -1):
        path[t] = trellis_state[(path[t + 1]), t + 1]

    if not ret_loglik:
        return path
    else:
        if scaled:
            loglik = -np.sum(np.log(scale))
        else:
            p = trellis_prob[path[-1], -1]
            loglik = np.log(p)
        return path, loglik


def log_sum_exp(x):
    """

    :param x:
    :return:
    """
    b = np.max(x, axis=1)
    sum_exp = np.sum(np.exp(x - b), axis=1)
    log_s_e = np.full(fill_value=np.finfo(float).min, shape=sum_exp.shape)
    np.log(sum_exp, where=(np.invert(np.isneginf(sum_exp))) & (sum_exp != 0.0), out=log_s_e)
    return b + log_s_e


def log_sum_exp_1d(x):
    """

    :param x:
    :return:
    """
    b = np.max(x)
    return b + np.log(np.sum(np.exp(x - b)))


def normalize(x):
    """

    :param x:
    :return:
    """
    return x / np.sum(x)


def log_softmax(x):
    """

    :param x:
    :return:
    """
    scale = log_sum_exp_1d(x)
    return x - scale, scale


def softmax(x):
    scale = np.sum(np.exp(x))
    return np.exp(x) / scale, scale
