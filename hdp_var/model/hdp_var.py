"""

"""
from collections import defaultdict

import numpy as np
from scipy.stats import invwishart, matrix_normal

from hdp_var.utils.HMM import compute_likelihoods, backwards_messaging, viterbi


class HDPVar:
    """
    HDP-SLDS model a re-organization of Emily Fox's Matlab code into python
    Only using the VAR and MNIW priors part
    for the sticky-HDP-HMM-VAR found on: https://homes.cs.washington.edu/~ebfox/software/
    This code uses the L - weak approximation to the DP
    therefore instead of adding states, it starts with L states and uses a subset of them,
     as long as L > actual number of states
    The following papers by Emily Fox should be read to understand the code:
        1) A STICKY HDP-HMM WITH APPLICATION TO SPEAKER DIARIZATION (2011)
        2) Bayesian Nonparametric Inference of Switching Dynamic Linear Models (2011)
    """

    def __init__(self, D, L, order):
        """

        :param D:
        :param int L: number of markov states in the HMM = number of SLDS modes
        :param order:
        """
        self.distributions = dict()
        # model parameters
        self.theta = {'A': np.empty((D, D * order, L)), 'Sigma': np.empty((D, D, L))}

        self.state_sequence = []
        # statistics of state occurrences
        """
        Parameter explanation:
        N - number of transitions - last row assigns 1 to z[0]
        bar_m - number of tables that "considered" ordering dish k in restaurant j 
        m - number of tables that were "served" dish k in restaurant j
        w - override variable for table t in restaurant j
        Ns - count of each state in the state sequence z
        """
        self.state_counts = {'N': np.zeros((L + 1, L)), 'bar_m': np.zeros((L, L)), 'm': np.zeros((L + 1, L)),
                             'w': np.zeros(L), 'Ns': np.zeros(L)}
        # maximum number of states
        self.L = L
        # VAR order
        self.order = order
        # number of dimensions
        self.D = D

        self.training_parameters = {'iterations': 1000, 'sample_every': 100, 'print_every': 100, 'burn_in': 100}
        self.sampling_parameters = {'K': np.linalg.inv(np.diag(0.5 * np.ones(self.D * self.order))),
                                    'M': np.zeros((self.D, self.D * self.order)), 'S_0': np.eye(self.D),
                                    'n_0': self.D + 2, 'a_gamma': 1, 'b_gamma': 0.001, 'a_alpha': 1,
                                    'b_alpha': 0.001, 'c': 100, 'd': 1}
        self.concentration_parameters = dict()
        self.param_tracking = defaultdict(list)

        # current training iteration
        self.iteration = 0
        # likelihoods of each state at each time point
        self.likelihoods = None
        self.total_log_likelihoods = []

        self.sample_concentration_parameters()

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

        :param params:
        :return:
        """
        self.sampling_parameters = {key: params.get(key,
                                                    self.sampling_parameters[key]) for key in
                                    self.sampling_parameters.keys()}
        self.sample_concentration_parameters()

    def sample_concentration_parameters(self):
        """

        :return:
        """

        self.concentration_parameters['gamma'] = self.sampling_parameters['a_gamma'] / \
                                                 self.sampling_parameters['b_gamma']
        self.concentration_parameters['alpha_p_kappa'] = self.sampling_parameters['a_alpha'] / \
                                                         self.sampling_parameters['b_alpha']
        self.concentration_parameters['rho'] = self.sampling_parameters['c'] / \
                                               (self.sampling_parameters['c'] + self.sampling_parameters['d'])

    def train(self, data, seed=None):
        """

        :param data:
        :param seed:
        :return:
        """
        iterations = np.arange(1, self.training_parameters['iterations'] + 1)
        if self.iteration != 0:
            iterations = np.arange(self.iteration + 1, self.training_parameters['iterations'])
        self.sample_init_theta(seed)
        self.sample_distributions(seed)
        for iteration in iterations:
            self.state_sequence, total_log_likelihood = self.sample_state_sequence(data, seed)
            self.sample_tables(seed)
            self.sample_distributions(seed)
            self.sample_theta(self.calculate_statistics(data))
            self.sample_hyperparameters(seed)

            if iteration != 1 and iteration % self.training_parameters['sample_every'] == 0 and\
                    iteration != self.training_parameters['iterations']:
                self.total_log_likelihoods.append(total_log_likelihood)
                self.param_tracking['state_sequence'].append(self.state_sequence)
                self.param_tracking['A'].append(self.theta['A'])
                self.param_tracking['Sigma'].append(self.theta['Sigma'])
                self.param_tracking['iteration'].append(iteration)
                self.param_tracking['gamma'].append(self.concentration_parameters['gamma'])
                self.param_tracking['alpha_p_kappa'].append(self.concentration_parameters['alpha_p_kappa'])
                self.param_tracking['rho'].append(self.concentration_parameters['rho'])
                self.param_tracking['transition_probabilities'].append(self.distributions['transition_probabilities'])
                self.param_tracking['initial_probabilities'].append(self.distributions['initial_probabilities'])
                self.param_tracking['beta'].append(self.distributions['beta'])
            if iteration != 1 and iteration % self.training_parameters['print_every'] == 0:
                print(f"Iteration: {iteration}/{self.training_parameters['iterations']}")
            self.iteration = iteration

        print(f'Training has stopped at iteration: {self.iteration}')
        params = self.get_best_iteration()
        return viterbi(self.L, data, theta={'A': params['A'], 'Sigma': params['Sigma']},
                       pi_0=params['initial_probabilities'], pi_z=params['transition_probabilities'])

    def sample_distributions(self, seed=None):
        """

        :param seed:
        :return:
        """
        # sample state, initial state and state transition probabilities
        kappa = self.concentration_parameters['rho'] * self.concentration_parameters['alpha_p_kappa']
        alpha = self.concentration_parameters['alpha_p_kappa'] - kappa

        rng = np.random.default_rng(seed=seed)
        # vector beta
        beta = rng.dirichlet(
            np.sum(self.state_counts['bar_m'] + self.concentration_parameters['gamma'] / self.L, axis=0))
        # transition probabilities
        pi_k = np.zeros((self.L, self.L))
        for k in range(self.L):
            # kronecker's alpha_p_kappa construction
            kron_apk = np.zeros(beta.shape)
            kron_apk[k] = 1
            pi_k[k] = rng.dirichlet(alpha * beta + self.state_counts['N'][k] + kappa * kron_apk)
        # initial probabilities
        pi_0 = rng.dirichlet(alpha * beta + self.state_counts['N'][self.L])
        self.distributions['transition_probabilities'] = pi_k
        self.distributions['initial_probabilities'] = pi_0
        self.distributions['beta'] = beta

    def sample_state_sequence(self, data, seed=None):
        """

        :param seed:
        :return:
        """
        self.likelihoods, total_log_likelihood = compute_likelihoods(self.L, data, self.theta,
                                                                     self.distributions['initial_probabilities'],
                                                                     self.distributions['transition_probabilities'])
        # number of observations
        T = np.shape(data['Y'])[1]
        block_size = data['block_size']
        block_end = data['block_end']

        part_marg_likelihood = backwards_messaging(size=(self.L, T),
                                                   pi_z=self.distributions['transition_probabilities'],
                                                   likelihoods=self.likelihoods)
        z = np.zeros(T, dtype=int)
        # forward run
        # pre-allocate indices
        assignment_info = [{'indices': np.zeros(T, dtype=int), 'total_count': 0} for k in range(self.L)]
        # tot_seq = np.zeros(self.L, dtype=int)
        # index_seq = np.zeros((T, self.L), dtype=int)
        Ns = np.zeros(self.state_counts['Ns'].shape, dtype=int)
        # transition probabilities
        N = np.zeros(self.state_counts['N'].shape, dtype=int)

        # initialize
        p_z = np.multiply(self.distributions['initial_probabilities'], part_marg_likelihood[:, 0])
        obs_inds = np.arange(block_end[0])
        # sampling from cdf
        z[0] = self.sample_state(p_z)
        # update state count
        N[-1, z[0]] += 1
        # count for block size
        for k in range(block_size[0]):
            Ns[z[0]] += 1
            # tot_seq[z[0]] += 1
            # index_seq[tot_seq[z[0]], z[0]] = obs_inds[k]

        for t in range(1, T):
            p_z = np.multiply(self.distributions['transition_probabilities'][z[t - 1]], part_marg_likelihood[:, t])
            # obs_inds = np.arange(block_end[t-1], block_end[t]+1)
            # sampling from cdf
            z[t] = HDPVar.sample_state(p_z)
            # update counts
            N[z[t-1], z[t]] += 1

            # count for block size
            for k in range(block_size[t]):
                Ns[z[t]] += 1
                # tot_seq[z[t]] += 1
                # index_seq[tot_seq[z[t]], z[t]] = obs_inds[k]

        # for k in range(self.L):
        #     # assignment_info[k]['total_count'] = tot_seq[k]
        #     assignment_info[k]['total_count'] =
        #     # assignment_info[k]['indices'] = index_seq[:, k]

        self.state_counts['Ns'] = Ns
        self.state_counts['N'] = N
        return z, total_log_likelihood
    
    @staticmethod
    def sample_state(pi_z_t):
        """
        Sample state i for time t (z[t] = i) using the inverse cdf method
        Update state counts in counts
        :param list(float) pi_z_t: state probabilities at time t = p(z[t]=i)
        :return:
        """
        cdf = np.cumsum(pi_z_t)
        return np.sum(cdf[-1] * np.random.rand(1) > cdf)

    def sample_tables(self, seed=None):
        """

        :param seed:
        :return:
        """
        np.random.seed(seed)
        # sample state, initial state and state transition probabilities
        kappa = self.concentration_parameters['rho'] * self.concentration_parameters['alpha_p_kappa']
        alpha = self.concentration_parameters['alpha_p_kappa'] - kappa

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
        rho = self.concentration_parameters['rho']
        p = 0.0
        w = np.zeros(self.state_counts['w'].shape)
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

    def sample_init_theta(self, seed=None):
        """

        :param seed:
        :return:
        """
        for k in range(self.L):
            Sigma = invwishart(df=self.sampling_parameters['n_0'] + self.state_counts['Ns'][k],
                               scale=self.sampling_parameters['S_0'], seed=seed).rvs()
            K_inv = np.linalg.cholesky(np.linalg.inv(self.sampling_parameters['K']))
            A = matrix_normal(mean=self.sampling_parameters['M'], rowcov=Sigma, colcov=K_inv, seed=seed).rvs()
            self.theta['A'][:, :, k] = A
            self.theta['Sigma'][:, :, k] = Sigma

    def sample_theta(self, data_statistics):
        """

        :param data_statistics: output of self.compute_statistics()
        :return:
        """
        S_xx, S_yy, S_yx = data_statistics
        for k in range(self.L):
            s_xx = S_xx[:, :, k] + self.sampling_parameters['K']
            s_yx = S_yx[:, :, k] + self.sampling_parameters['M'].dot(self.sampling_parameters['K'])
            s_yy = S_yy[:, :, k] + self.sampling_parameters['M'].dot(
                self.sampling_parameters['K'].dot(np.transpose(self.sampling_parameters['M'])))
            s_yx_s_xx_inv = s_yx.dot(np.linalg.inv(s_xx))
            s_ygx = s_yy - s_yx_s_xx_inv.dot(np.transpose(s_yx))
            # enforce symmetry
            s_ygx = (s_ygx + np.transpose(s_ygx)) / 2
            Sigma = invwishart(df=self.sampling_parameters['n_0'] + self.state_counts['Ns'][k],
                               scale=self.sampling_parameters['S_0'] + s_ygx).rvs()
            K_inv = np.linalg.cholesky(np.linalg.inv(self.sampling_parameters['K']))
            A = matrix_normal(mean=s_yx_s_xx_inv, rowcov=Sigma, colcov=K_inv).rvs()
            self.theta['A'][:, :, k] = A
            self.theta['Sigma'][:, :, k] = Sigma

    def sample_hyperparameters(self, seed=None):
        """

        :param seed:
        :return:
        """
        rng = np.random.default_rng(seed)
        n_dot = np.sum(self.state_counts['N'], axis=1)
        m_dot = np.sum(self.state_counts['m'], axis=1)
        bar_k = np.sum(np.sum(self.state_counts['bar_m'], axis=0) > 0)
        valid_indices = np.where(n_dot > 0)
        self.concentration_parameters['rho'] = rng.beta(self.sampling_parameters['c'] + np.sum(self.state_counts['w']),
                                                        self.sampling_parameters['d'] + np.sum(self.state_counts['m']) -
                                                        np.sum(self.state_counts['w']))
        if valid_indices:
            self.concentration_parameters['alpha_p_kappa'] = rng.gamma(self.sampling_parameters['a_alpha'] /
                                                                       self.sampling_parameters['b_alpha'])
            self.concentration_parameters['gamma'] = rng.gamma(self.sampling_parameters['a_gamma'] /
                                                               self.sampling_parameters['b_gamma'])
            return
        self.concentration_parameters['alpha_p_kappa'] = self.resample_conc_parameter(
            conc_param=self.concentration_parameters['alpha_p_kappa'],
            table_customer_count=n_dot[valid_indices],
            table_count=m_dot[valid_indices],
            a=self.sampling_parameters['a_alpha'],
            b=self.sampling_parameters['b_alpha'],
            iterations=50, seed=seed)
        self.concentration_parameters['gamma'] = self.resample_conc_parameter(
            conc_param=self.concentration_parameters['gamma'],
            table_customer_count=np.sum(
                self.state_counts['bar_m']),
            table_count=bar_k,
            a=self.sampling_parameters['a_gamma'],
            b=self.sampling_parameters['b_gamma'],
            iterations=50, seed=seed)

    def resample_conc_parameter(self, conc_param, table_customer_count, table_count, a, b, iterations, seed=None):
        """

        :param conc_param:
        :param table_customer_count:
        :param table_count:
        :param a:
        :param b:
        :param iterations:
        :param seed:
        :return:
        """
        # auxiliary variable resampling of DP concentration parameter
        rng = np.random.default_rng(seed)
        restaurant_count = len(table_customer_count)
        # table count in all restaurants together
        total_table_count = np.sum(table_count)

        A = np.array(list(zip(conc_param + 1, table_customer_count))).transpose()

        for i in range(iterations):
            # beta auxilary variables
            beta_aux = rng.dirichlet(A)[0]
            # binomial auxilary variables
            binom_aux = np.multiply(np.random.rand(restaurant_count),
                                    conc_param + table_customer_count) < table_customer_count
            # gamma resampling of concentration parameter
            gamma_a = a + total_table_count - np.sum(binom_aux)
            gamma_b = b - np.sum(np.log(beta_aux))
            conc_param = rng.gamma(gamma_a) / gamma_b
        return conc_param

    def get_best_iteration(self):
        """

        :return:
        """
        argmax = np.argmax(self.total_log_likelihoods)
        params = {}
        for key in self.param_tracking.keys():
            params[key] = self.param_tracking[key][argmax]
        return params
