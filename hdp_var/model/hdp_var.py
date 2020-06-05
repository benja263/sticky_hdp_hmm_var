"""

"""
import numpy as np
from collections import defaultdict
from scipy.stats import invwishart, matrix_normal


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
        self.theta = {'A': np.zeros((D, D * order, L)), 'Sigma': np.zeros((D, D, L))}

        self.state_sequence = []
        # statistics of state occurrences
        """
        Parameter explanation:
        N - number of transitions - last row assigns 1 to z[0]
        barM - number of tables that "considered" ordering dish k in restaurant j 
        m - number of tables that were "served" dish k in restaurant j
        w - override variable for table t in restaurant j
        Ns - count of each state in the state sequence z
        """
        self.state_counts = {'N': np.zeros((L + 1, L)), 'barM': np.zeros((L, L)), 'm': np.zeros((L, L)),
                             'w': np.zeros((L, 1)), 'Ns': np.zeros((L, 1))}
        # maximum number of states
        self.L = L
        # VAR order
        self.order = order
        # number of dimensions
        self.D = D

        self.training_parameters = {'iterations': 1000, 'save_every': 100, 'print_every': 100, 'burn_in': 100}
        self.sampling_parameters = {'K': np.linalg.inv(np.diag(0.5 * np.ones(self.D * self.order))),
                                    'M': np.zeros((self.D, self.D * self.order)), 'S_0': np.eye(self.D),
                                    'n_0': self.D + 2, 'a_gamma': 1, 'b_gamma': 0.001, 'a_alpha': 1,
                                    'b_alpha': 0.001, 'c': 100, 'd': 1}
        self.concentration_parameters = dict()
        self.param_tracking = defaultdict(list)

        # current training iteration
        self.iteration = 0
        # likelihoods of each state at each time point
        self.likelihoods = []

        self.sample_concentration_parameters()

    def set_training_parameters(self, params):
        """

        :param params: training parameters
        :return:
        """
        self.training_parameters = {key: params.get(key,
                                                    self.training_parameters[key]) for key in
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
            np.sum(self.state_counts['barM'] + self.concentration_parameters['gamma'] / self.L, axis=0))
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
        np.random.seed(seed)
        total_log_likelihood = self.compute_likelihoods(data)
        # number of observations
        T = np.shape(data['observations'])[1]
        block_size = data['block_size']
        block_end = data['block_end']

        _, part_marg_likelihood = self.backwards_messaging()
        z = np.zeros(T)
        # forward run
        # pre-allocate indices
        INDS = {'obsIndsz': [{'inds': np.zeros(T), 'tot': 0} for i in range(self.L)]}
        totSeq = np.zeros(self.L)
        indSeq = np.zeros(T, self.L)
        Ns = np.zeros(self.state_counts['Ns'].shape)
        # transition probabilities
        N = np.zeros(self.state_counts['N'].shape)

        # initialize
        p_z = self.distributions['initial_probabilities'] * part_marg_likelihood[:, 1]
        obsInd = np.arange(block_end[0])
        p_z = np.cumsum(p_z)
        z[0] = 1 + np.sum(p_z[-1] * np.random.rand(1) > p_z)

        N[-1, z[0]] = 1

        # count for block size
        for k in range(block_size[0]):
            Ns[z[0]] += 1
            totSeq[z[0]] += 1
            indSeq[totSeq[z[0]], z[0]] = obsInd[k]

        for t in range(1, T):
            p_z = self.distributions['transition_probabilities'][z[t - 1]] * part_marg_likelihood[:, 1]
            obsInd = np.arange(block_end[0])
            p_z = np.cumsum(p_z)
            # todo consider removing 1 as it may be related to index starting at 1 in matlab but i am not sure
            z[t] = 1 + np.sum(p_z[-1] * np.random.rand(1) > p_z)
            N[z[t - 1], z[t]] += 1

            # count for block size
            for k in range(block_size[t]):
                Ns[z[t]] += 1
                totSeq[z[t]] += 1
                indSeq[totSeq[z[t]], z[t]] = obsInd[k]

        for k in range(self.L):
            INDS['obsIndsz'][k]['tot'] = totSeq[k]
            INDS['obsIndsz'][k]['inds'] = indSeq[:, k]

        self.state_counts['Ns'] = Ns
        self.state_counts['N'] = N
        return z, INDS, total_log_likelihood

    def compute_likelihoods(self, data):
        """

        :param data:
        :return:
        """
        D, T = np.shape(data['observations'])
        log_likelihood = np.zeros(self.L, T)
        Y = data['observations']
        X = data['X']

        for k in range(self.L):
            chol_inv_sigma = np.linalg.cholesky(self.theta['Sigma'][:, :, k])
            dchol_inv_sigma = np.diag(chol_inv_sigma)
            mu = chol_inv_sigma * (Y - self.theta['A'][:, :, k] * X)
            log_likelihood[k] = -0.5 * np.sum(mu ** 2, axis=0) + np.sum(np.log(dchol_inv_sigma))
        # todo how to do this
        normalizer = np.max(log_likelihood, axis=1)
        log_likelihood -= normalizer

        self.likelihoods = np.exp(log_likelihood)
        normalizer -= (D / 2) * np.log(2 * np.pi)

        # forward pass to integrate over the state sequence
        fwd_msg = np.zeros((self.L, T))
        fwd_msg[:, 1] = np.multiply(self.likelihoods[:, 1], self.distributions['initial_probabilities'])
        # normalize
        # todo look at the correct summing axis
        sum_fwd_msg = np.sum(fwd_msg, axis=0)
        fwd_msg[:, 1] /= sum_fwd_msg

        # add the constant for normalizing the forward message
        # murphy's book page 611 Z is the constant
        normalizer[0] += np.log(sum_fwd_msg)
        # compute messages forward in time
        for t in range(T - 1):
            # integrate out z[t]
            partial_marg_likelihood = self.distributions['transition_probabilities'] * fwd_msg[:, t]
            # multiply likelihood by incoming message
            fwd_msg[:, t + 1] = np.multiply(partial_marg_likelihood, self.likelihoods[:, t + 1])
            sum_fwd_msg = np.sum(fwd_msg[:, t + 1])
            fwd_msg[:, t + 1] /= sum_fwd_msg
            normalizer[t + 1] += np.log(sum_fwd_msg)
        # total log-likelihood
        return np.sum(normalizer)

    def backwards_messaging(self, T):
        """

        :param T:
        :return:
        """
        back_msg = np.zeros((self.L, T))
        back_msg[:, -1] = np.ones(self.L)
        partial_marg_likelihood = np.zeros(back_msg.shape)
        for t in range(start=T - 1, stop=0, step=-1):
            # multiply likelihood by incoming message
            partial_marg_likelihood[:, t + 1] = np.multiply(self.likelihoods[:, t + 1], back_msg[:, t + 1])
            # integrate over z(t)
            back_msg[:, t] = np.multiply(self.distributions['transition_probabilities'],
                                         partial_marg_likelihood[:, t + 1])
            # normalize
            back_msg[:, t] /= np.sum(back_msg[:, t])
        # for t=0
        # todo check if needed
        partial_marg_likelihood[:, 0] = np.multiply(self.likelihoods[:, 0], back_msg[:, 0])
        return back_msg, partial_marg_likelihood

    def sample_tables(self, seed=None):
        """

        :param seed:
        :return:
        """
        np.random.seed(seed)
        # sample state, initial state and state transition probabilities
        kappa = self.sampling_parameters['rho'] * self.sampling_parameters['alpha_p_kappa']
        alpha = self.sampling_parameters['alpha_p_kappa'] - kappa

        m = np.empty(self.state_counts['N'].shape)
        a_beta = alpha * self.distributions['beta']
        # sample M where M[i,j] = # of tables in restaurant i that served dish j
        vec = a_beta * np.ones(self.L) + kappa * np.eye(self.L)
        vec = np.concatenate((vec, a_beta[:, np.newaxis].transpose()))
        N = self.state_counts['N'].ravel()
        for ind, element in enumerate(vec.ravel()):
            temp = 1 + np.sum(np.random.rand(1, N[ind] - 1) < np.ones((1, N[ind] - 1))) * element
            m[ind] = np.divide(temp, element + np.arange(1, N[ind] + 1))
        m[self.state_counts['N'] == 0.0] = 0.0
        self.state_counts['m'] = m
        self.sample_barM()

    def sample_barM(self):
        """

        :return:
        """
        barM = self.state_counts['m'].copy()
        rho = self.sampling_parameters['rho']
        p = 0.0
        w = np.zeros(self.state_counts['w'].shape)
        for i in range(barM.shape[1]):
            if rho > 0.0:
                p = rho / (self.distributions['beta'][i] * (1 - rho) + rho)
            w[i] = np.random.binomial(p, self.state_counts['m'][i, i])
            barM[i, i] -= w
        self.state_counts['barM'] = barM
        self.state_counts['w'] = w

    def calculate_statistics(self, data, INDS):
        """

        :param INDS:
        :return:
        """
        dim_obs, dim_x = data['observations'].shape[0], data['X'].shape[0]
        S_xx, S_yy = np.zeros((dim_x, dim_x, self.L)), np.zeros((dim_obs, dim_obs, self.L))
        S_yx = np.zeros((dim_obs, dim_x, self.L))

        obs, X = data['observations'], data['X']
        for k in range(self.L):
            # todo see if this works
            obsInd = INDS['obsIndzs'][k]['inds'][np.arange(INDS['obsIndzs'][k]['tot'])]
            # todo check if it is element wise or dot product
            S_xx[:, :, k] += X[:, obsInd].dot(X[:, obsInd])
            S_yy[:, :, k] += obs[:, obsInd].dot(obs[:, obsInd])
            S_yx[:, :, k] += obs[:, obsInd].dot(X[:, obsInd])
        return S_xx, S_yy, S_yx

    def sample_init_theta(self, seed=None):
        """

        :param seed:
        :return:
        """
        rng = np.random.default_rng(seed=seed)
        for k in range(self.L):
            Sigma = invwishart(df=self.sampling_parameters['n_0'] + self.state_counts['Ns'][k],
                               scale=self.sampling_parameters['S_0'], random_state=rng)
            K_inv = np.linalg.cholesky(np.linalg.inv(self.sampling_parameters['K']))
            A = matrix_normal(mean=self.sampling_parameters['M'], rowcov=Sigma, colcov=K_inv, random_state=rng)
            self.theta['A'][:, :, k] = A
            self.theta['Sigma'][:, :, k] = Sigma

    def sample_theta(self, INDS, seed=None):
        """

        :param INDS:
        :param seed:
        :return:
        """
        rng = np.random.default_rng(seed=seed)
        S_xx, S_yy, S_yx = self.calculate_statistics(INDS)
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
                               scale=self.sampling_parameters['S_0'] + s_ygx, random_state=rng)
            K_inv = np.linalg.cholesky(np.linalg.inv(self.sampling_parameters['K']))
            A = matrix_normal(mean=s_yx_s_xx_inv, rowcov=Sigma, colcov=K_inv, random_state=rng)
            self.theta['A'][:, :, k] = A
            self.theta['Sigma'][:, :, k] = Sigma


    def sample_hyperparameters(self, seed=None):
        """

        :param seed:
        :return:
        """
        np.random.seed(seed)
        n_dot = np.sum(self.state_counts['N'], axis=1)
        m_dot = np.sum(self.state_counts['m'], axis=1)

    def gibs_conparam(self):
        # auxiliary variable resampling of DP concentration parameter
        pass

    def generate_data_structure(self, observations):
        """

        :param observations:
        :return:
        """
        X, valid = self.make_design_matrix(observations, self.order)
        data = {'observations': observations[:, valid], 'X': X[:, valid]}
        data['blockSize'] = np.ones((1, data['observations'].shape[1])),
        data['blockEnd'] = np.cumsum(data['blockSize'])
        return data

    def viterbi(self, data):
        """

        :return:
        """
        log_likelihood = self.compute_likelihoods(data)
        T = log_likelihood.shape[1]
        delta, psi = np.zeros(log_likelihood.shape), np.zeros(log_likelihood.shape)
        path = np.zeros(T)
        scale = np.ones(T)

        delta[:, 0] = np.multiply(self.distributions['initial_probabilities'], self.likelihoods[:, 0])
        scale[0] = np.sum(delta[:, 0])
        delta[:, 0] /= scale[0]

        for t in range(start=1, stop=T):
            for j in range(self.L):
                # todo invented partial marg name - need to see what it means
                partial_marg = np.multiply(delta[:, t - 1], self.distributions['transition_probabilities'][:, j])
                delta[j, t], psi[j, t] = np.max(partial_marg) * self.likelihoods[j, t], np.argmax(partial_marg)
            scale[t] = np.sum(delta[:, t])
            delta[:, t] /= scale[t]

        path[T] = np.argmax(delta[:, T])
        for t in range(start=T - 1, stop=0, step=-1):
            path[t] = psi[path[t + 1], t + 1]
        self.state_sequence = path

    def update_state_counts(self):
        """

        :return:
        """

    def make_design_matrix(self, obs, order):
        pass

    def find_best_iteration(self):
        """

        :return:
        """
        argmax = np.argmax(self.total_log_likelihood)
        self.sampling_parameters['gamma'] = self.param_tracking['gamma'][argmax]
        self.sampling_parameters['alpha_p_kappa'] = self.param_tracking['alpha_p_kappa'][argmax]
        self.sampling_parameters['rho'] = self.param_tracking['rho'][argmax]

        self.theta['A'] = self.param_tracking['A'][argmax]
        self.theta['Sigma'] = self.param_tracking['Sigma'][argmax]

        self.distributions['beta'] = self.param_tracking['beta'][argmax]
        self.distributions['transition_probabilities'] = self.param_tracking['transition_probabilities'][argmax]
        self.distributions['initial_probabilities'] = self.param_tracking['initial_probabilities'][argmax]
