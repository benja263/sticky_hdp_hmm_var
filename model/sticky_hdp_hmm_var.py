"""
Module containing the HDPVar model
"""
from collections import defaultdict

import numpy as np
from scipy.linalg import inv
from scipy.stats import invwishart, matrix_normal

from utils.HMM import compute_likelihoods, backwards_messaging, viterbi
from utils.math.c_extensions import load_c_lib, rand_gamma, rand_dirichlet


class HDPVar:
    """
    sticky-HDP-HMM-SLDS model Only using the sticky-VAR and MNIW priors part
    This code uses the L - weak approximation to the DP
    therefore instead of adding states, it starts with L states and uses a subset of them,
     as long as L > actual number of states
    The following papers by Emily Fox should be read to understand the code:
        1) A STICKY HDP-HMM WITH APPLICATION TO SPEAKER DIARIZATION (2011)
        2) Bayesian Nonparametric Inference of Switching Dynamic Linear Models (2011)
    """

    def __init__(self, D, L, order):
        """
        State Count Parameter Explanation:
        N - number of transitions - last row assigns 1 to z[0]
        bar_m - number of tables that "considered" ordering dish k in restaurant j
        m - number of tables that were "served" dish k in restaurant j
        w - override variable for table t in restaurant j
        Ns - count of each state in the state sequence z
        :param D:
        :param int L: number of markov states in the HMM = number of SLDS modes
        :param order:
        """
        self.distributions = dict()
        # model parameters
        self.theta = {'A': np.ones((D, D * order, L)), 'inv_sigma': np.ones((D, D, L))}

        self.state_sequence = []
        # statistics of state occurrences
        self.state_counts = {'N': np.zeros((L + 1, L)), 'bar_m': np.zeros((L, L)), 'm': np.zeros((L + 1, L)),
                             'w': np.zeros(L), 'Ns': np.zeros(L)}
        # maximum number of states
        self.L = L
        # VAR order
        self.order = order
        # number of dimensions
        self.D = D

        self.training_parameters = {'iterations': 1000, 'sample_every': 15, 'print_every': 100, 'burn_in': 100}
        self.sampling_parameters = {'K': inv(np.diag(0.5 * np.ones(self.D * self.order))),
                                    'M': np.zeros((self.D, self.D * self.order)), 'S_0': np.eye(self.D),
                                    'n_0': self.D + 2, 'a_gamma': 1, 'b_gamma': 0.001, 'a_alpha': 1,
                                    'b_alpha': 0.001, 'c': 100, 'd': 1}
        self.DP_parameters = dict()
        self.param_tracking = defaultdict(list)
        # current training iteration
        self.iteration = 0
        # likelihoods of each state at each time point
        self.sequence_log_likelihoods = []
        self.init_DP_parameters()
        self.c_lib = load_c_lib()

    def set_training_parameters(self, params):
        """

        :param params: training parameters
        :return:
        """
        self.training_parameters = {key: params.get(key,
                                                    self.training_parameters[key]) if params.get(key, None) is not None
        else self.training_parameters[key] for key in
                                    self.training_parameters.keys()}

    def set_sampling_parameters(self, params):
        """

        :param params: training parameters
        :return:
        """
        self.sampling_parameters = {key: params.get(key,
                                                    self.sampling_parameters[key]) if params.get(key, None) is not None
        else self.sampling_parameters[key] for key in
                                    self.sampling_parameters.keys()}
        self.init_DP_parameters()

    def init_DP_parameters(self):
        self.DP_parameters['gamma'] = self.sampling_parameters['a_gamma'] / self.sampling_parameters['b_gamma']
        self.DP_parameters['alpha_p_kappa'] = self.sampling_parameters['a_alpha'] / self.sampling_parameters['b_alpha']
        self.DP_parameters['rho'] = self.sampling_parameters['c'] / (self.sampling_parameters['c'] +
                                                                     self.sampling_parameters['d'])

    def train(self, data, seed=None):
        iterations = np.arange(1, self.training_parameters['iterations'] + 1)
        if self.iteration != 0:
            iterations = np.arange(self.iteration + 1, self.training_parameters['iterations'])
            self.c_lib = load_c_lib()
        self.sample_distributions(seed)
        self.sample_init_theta(seed)
        for iteration in iterations:
            self.state_sequence, sequence_log_likelihood = self.sample_state_sequence(data)
            self.sample_tables(seed)
            self.sample_distributions(seed)
            self.sample_theta(self.calculate_statistics(data))
            self.sample_hyperparameters(seed)

            if iteration >= self.training_parameters['burn_in'] and iteration % self.training_parameters['sample_every'] == 0:
                self.sequence_log_likelihoods.append(sequence_log_likelihood)
                self.param_tracking['state_sequence'].append(self.state_sequence)
                self.param_tracking['A'].append(self.theta['A'])
                self.param_tracking['inv_sigma'].append(self.theta['inv_sigma'])
                self.param_tracking['iteration'].append(iteration)
                self.param_tracking['gamma'].append(self.DP_parameters['gamma'])
                self.param_tracking['alpha_p_kappa'].append(self.DP_parameters['alpha_p_kappa'])
                self.param_tracking['rho'].append(self.DP_parameters['rho'])
                self.param_tracking['pi_z'].append(self.distributions['pi_z'])
                self.param_tracking['pi_0'].append(self.distributions['pi_0'])
                self.param_tracking['beta'].append(self.distributions['beta'])
            if iteration % self.training_parameters['print_every'] == 0:
                print(f"Iteration: {iteration}/{self.training_parameters['iterations']}")
                print(f'Sequence log-likelihood: {sequence_log_likelihood}')
            self.iteration = iteration
        self.c_lib = None
        print(f'Training has stopped at iteration: {self.iteration}')

    def sample_distributions(self, seed=None):
        """

        :param seed:
        :return:
        """
        # sample state, initial state and state transition probabilities
        kappa = self.DP_parameters['alpha_p_kappa'] * self.DP_parameters['rho']
        alpha = self.DP_parameters['alpha_p_kappa'] - kappa
        # vector beta
        beta = rand_dirichlet(
            (np.sum(self.state_counts['bar_m'], axis=0) + self.DP_parameters['gamma'] / self.L).flatten(),
            self.c_lib)
        pi_z = np.zeros((self.L, self.L))
        for k in range(self.L):
            # kronecker's alpha_p_kappa construction
            kron_apk = np.zeros(beta.shape)
            kron_apk[k] = 1
            pi_z[k] = rand_dirichlet((alpha * beta + self.state_counts['N'][k] + kappa * kron_apk).flatten(),
                                     self.c_lib)
        # initial probabilities
        pi_0 = rand_dirichlet((alpha * beta + self.state_counts['N'][self.L]).flatten(), self.c_lib)
        self.distributions['pi_z'] = pi_z
        self.distributions['pi_0'] = pi_0[np.newaxis, :]
        self.distributions['beta'] = beta

    def sample_init_theta(self, seed=None):
        for k in range(self.L):
            sigma = invwishart(df=self.sampling_parameters['n_0'] + self.state_counts['Ns'][k],
                               scale=self.sampling_parameters['S_0'] + 0, seed=seed).rvs()
            A = matrix_normal(mean=self.sampling_parameters['M'], rowcov=sigma,
                              colcov=inv(self.sampling_parameters['K'])).rvs()
            self.theta['A'][:, :, k] = A
            self.theta['inv_sigma'][:, :, k] = inv(sigma)

    def sample_state_sequence(self, data):
        pi_0, pi_z = self.distributions['pi_0'], self.distributions['pi_z']
        likelihoods, log_likelihoods, sequence_log_likelihood = compute_likelihoods(self.L, data,
                                                                                                  self.theta,
                                                                                                  self.distributions[
                                                                                                      'pi_0'],
                                                                                                  self.distributions[
                                                                                                      'pi_z'],
                                                                                                  self.c_lib)
        # number of observations
        T = np.shape(data['Y'])[1]
        back_msg = backwards_messaging(size=(self.L, T), pi_z=pi_z, log_likelihoods=log_likelihoods, c_lib=self.c_lib)
        z = np.zeros(T, dtype=int)
        # forward run
        # pre-allocate indices
        Ns = np.zeros(self.state_counts['Ns'].shape, dtype=int)
        # transition probabilities
        N = np.zeros(self.state_counts['N'].shape, dtype=int)
        # initialize
        f = likelihoods[:, 0] * back_msg[:, 0]
        # normalize density
        f /= np.sum(f)
        state_likelihood = pi_0 * f
        # sampling from cdf
        z[0] = self.sample_state(state_likelihood / np.sum(state_likelihood))
        # update state count
        N[-1, z[0]] += 1
        Ns[z[0]] += 1
        for t in range(1, T):
            f = likelihoods[:, t] * back_msg[:, t]
            f /= np.sum(f)
            state_likelihood = pi_z[z[t - 1]] * f
            z[t] = self.sample_state(state_likelihood / np.sum(state_likelihood))
            # update counts
            N[z[t - 1], z[t]] += 1
            Ns[z[t]] += 1
        self.state_counts['Ns'] = Ns
        self.state_counts['N'] = N
        return z, sequence_log_likelihood

    @staticmethod
    def sample_state(pi_z_t):
        """
        Sample state i for time t (z[t] = i) using the inverse cdf method
        Update state counts in counts
        :param list(float) pi_z_t: state probabilities at time t = p(z[t]=i)
        :return:
        """
        cdf = np.cumsum(pi_z_t)
        return np.sum(cdf[-1] * np.random.rand() > cdf)

    def sample_tables(self, seed=None):
        """

        :param seed:
        :return:
        """
        np.random.seed(seed)
        # sample state, initial state and state transition probabilities
        kappa = self.DP_parameters['alpha_p_kappa'] * self.DP_parameters['rho']
        alpha = self.DP_parameters['alpha_p_kappa'] - kappa

        m = np.zeros(self.state_counts['N'].shape, dtype=int).ravel()
        a_beta = alpha * self.distributions['beta']
        # sample M where M[i,j] = # of tables in restaurant i that served dish j
        vec = a_beta * np.ones(self.L) + kappa * np.eye(self.L)
        vec = np.concatenate((vec, a_beta[:, np.newaxis].transpose()))
        N = self.state_counts['N'].ravel()
        for ind, element in enumerate(vec.ravel()):
            m[ind] = 1 + np.sum(np.random.rand(1, N[ind]) < np.divide(np.ones(N[ind]) * element,
                                                                      element + np.arange(N[ind])))
        m[N == 0.0] = 0.0
        self.state_counts['m'] = m.reshape(self.state_counts['m'].shape)
        self.sample_bar_m()

    def sample_bar_m(self):
        """

        :return:
        """
        bar_m = self.state_counts['m'].copy()
        rho = self.DP_parameters['rho']
        p, w = 0.0, np.zeros(self.state_counts['w'].shape)
        for i in range(bar_m.shape[1]):
            if rho > 0.0:
                p = rho / (self.distributions['beta'][i] * (1 - rho) + rho)
            w[i] = np.random.binomial(self.state_counts['m'][i, i], p)
            bar_m[i, i] -= w[i]
        self.state_counts['bar_m'] = bar_m
        self.state_counts['w'] = w

    def calculate_statistics(self, data):
        """

        :param data:
        :return:
        """
        dim_y, dim_x = data['Y'].shape[0], data['X'].shape[0]
        S_xx, S_yy = np.zeros((dim_x, dim_x, self.L)), np.zeros((dim_y, dim_y, self.L))
        S_yx = np.zeros((dim_y, dim_x, self.L))

        for k in range(self.L):
            S_xx[:, :, k] += data['X'][:, self.state_sequence == k].dot(
                data['X'][:, self.state_sequence == k].transpose())
            S_yy[:, :, k] += data['Y'][:, self.state_sequence == k].dot(
                data['Y'][:, self.state_sequence == k].transpose())
            S_yx[:, :, k] += data['Y'][:, self.state_sequence == k].dot(
                data['X'][:, self.state_sequence == k].transpose())
        return S_xx, S_yy, S_yx

    def sample_theta(self, data_statistics):
        """

        :param data_statistics: output of self.compute_statistics()
        :return:
        """
        S_xx, S_yy, S_yx = data_statistics
        for k in range(self.L):
            s_xx = S_xx[:, :, k] + self.sampling_parameters['K']
            s_yx = S_yx[:, :, k] + self.sampling_parameters['M'] @ self.sampling_parameters['K']
            s_yy = S_yy[:, :, k] + self.sampling_parameters['M'] @ (
                    self.sampling_parameters['K'] @ self.sampling_parameters['M'].T)
            M = s_yx @ inv(s_xx)
            s_ygx = s_yy - M @ s_yx.T
            # enforce symmetry
            s_ygx = (s_ygx + s_ygx.T) / 2
            sigma = invwishart(df=self.sampling_parameters['n_0'] + self.state_counts['Ns'][k],
                               scale=self.sampling_parameters['S_0'] + s_ygx).rvs()
            A = matrix_normal(mean=M + self.sampling_parameters['M'], rowcov=sigma, colcov=inv(s_xx)).rvs()
            self.theta['A'][:, :, k] = A
            self.theta['inv_sigma'][:, :, k] = inv(sigma)

    def sample_hyperparameters(self, seed=None):
        rng = np.random.default_rng(seed)
        n_dot = np.sum(self.state_counts['N'], axis=1)
        m_dot = np.sum(self.state_counts['m'], axis=1)
        bar_k = np.sum(np.sum(self.state_counts['bar_m'], axis=0) > 0)
        valid_indices = np.where(n_dot > 0)[0].tolist()
        self.DP_parameters['rho'] = rng.beta(self.sampling_parameters['c'] + np.sum(self.state_counts['w']),
                                             self.sampling_parameters['d'] + np.sum(self.state_counts['m']) -
                                             np.sum(self.state_counts['w']))
        if not valid_indices:
            self.DP_parameters['alpha_p_kappa'] = rng.gamma(self.sampling_parameters['a_alpha']) / \
                                                  self.sampling_parameters['b_alpha']
            self.DP_parameters['gamma'] = rand_gamma(np.array([self.sampling_parameters['a_gamma'] / \
                                                               self.sampling_parameters['b_gamma']]).flatten(),
                                                     self.c_lib)
            return
        self.DP_parameters['alpha_p_kappa'] = self.resample_DP_parameter(
            DP_param=self.DP_parameters['alpha_p_kappa'],
            table_customer_count=n_dot[valid_indices],
            table_count=m_dot[valid_indices],
            a=self.sampling_parameters['a_alpha'],
            b=self.sampling_parameters['b_alpha'],
            iterations=50)
        self.DP_parameters['gamma'] = self.resample_DP_parameter(
            DP_param=self.DP_parameters['gamma'],
            table_customer_count=np.sum(
                self.state_counts['bar_m']),
            table_count=bar_k,
            a=self.sampling_parameters['a_gamma'],
            b=self.sampling_parameters['b_gamma'],
            iterations=50)

    def resample_DP_parameter(self, DP_param, table_customer_count, table_count, a, b, iterations):
        # auxiliary variable resampling of DP concentration parameter
        restaurant_count = table_customer_count.size
        if not isinstance(table_customer_count, np.ndarray):
            restaurant_count = 1
        # table count in all restaurants together
        total_table_count = np.sum(table_count)
        A = np.zeros((restaurant_count, 2))
        A[:, 0], A[:, 1] = DP_param + 1, table_customer_count
        A = A.transpose()

        for i in range(iterations):
            # beta auxiliary variables
            beta_aux = np.zeros(A.shape)
            for j in range(beta_aux.shape[1]):
                beta_aux[:, j] = rand_dirichlet((A[:, j]).flatten(), self.c_lib)
            # binomial auxiliary variables
            binom_aux = np.random.rand(restaurant_count) * (DP_param + table_customer_count) < table_customer_count
            # gamma resampling of concentration parameter
            gamma_a, gamma_b = a + total_table_count - np.sum(binom_aux), b - np.sum(np.log(beta_aux[0]))
            DP_param = rand_gamma(np.array([gamma_a / gamma_b]).flatten(), self.c_lib)
        return DP_param

    def get_best_iteration(self):
        """
        Return model parameters corresponding to the sample with the highest log-likelihood
        :return:
        """
        argmax = int(np.argmax(self.sequence_log_likelihoods))
        params = {}
        for key in self.param_tracking.keys():
            params[key] = self.param_tracking[key][argmax]
        return params

    def predict_state_sequence(self, data):
        """
        Predict HMM state-sequence using the viterbi algorithm
        :param data:
        :return:
        """
        params = self.get_best_iteration()
        self.state_sequence = viterbi(self.L, data, theta={'A': params['A'], 'inv_sigma': params['inv_sigma']},
                                      pi_0=params['pi_0'], pi_z=params['pi_z'], c_lib=self.c_lib or load_c_lib())
        self.theta['A'], self.theta['inv_sigma'] = params['A'], params['inv_sigma']
        return self.state_sequence

    def predict_observations(self, X_0, reset_every=None):
        """
        Predict observations of the type Y[t] = A[t-1]*Y[t-1] + A[t-2]*Y[t-2] + ... + A[t-r]*Y[t-r].
        As errors in time-series accumulate one may "reset" the prediction error by using the ground truth past observations
        X_0 every x number of time points.
        Note - first predict state sequence prior to predicting observations.
        :param X_0: represents the ground truth past observations such that X_0[:, 0] = the prior observations that
        generated Y[:, 0]
        :param int reset_every: Reset every x time points
        :return: DXT np.ndarray where D = number of dimensions and T = number of time points
        """
        T = len(self.state_sequence)
        pred_Y, z = np.zeros((self.D, T)), self.state_sequence
        A, t = self.theta['A'], 0
        while t < T:
            if t < self.order or (reset_every is not None and t % reset_every == 0) and T - t > reset_every:
                for r in range(self.order):
                    pred_Y[:, t + r] = A[:, :, z[t]] @ X_0[:, t + r]
                t += self.order
            else:
                X_pred = pred_Y[:, t - 1]
                for r in range(2, self.order + 1):
                    X_pred = np.concatenate((X_pred, pred_Y[:, t - r]), axis=0)
                pred_Y[:, t] = A[:, :, z[t]] @ X_pred
                t += 1
        return pred_Y
