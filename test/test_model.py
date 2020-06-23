"""

"""
import unittest
import numpy as np
from hdp_var.model.hdp_var import HDPVar
from hdp_var.parameters import SamplingParams, TrainingParams


class TestModel(unittest.TestCase):
    """

    """
    order = 2
    D = 2
    L = 2

    def test_init(self):
        """

        :return:
        """
        model = HDPVar(order=self.order, D=self.D, L=self.L)
        self.assertEqual(model.D, self.D, msg='D value != model D value')
        self.assertEqual(model.order, self.order, msg='order value != model order value')
        self.assertEqual(model.L, self.L, msg='L value != model L value')

        self.assertEqual(model.sampling_parameters['n_0'], self.D + 2, msg='n_0 != default value')
        self.assertEqual(model.sampling_parameters['a_alpha'], 1, msg='a_alpha != default value')
        self.assertEqual(model.sampling_parameters['b_alpha'], 0.001, msg='b_alpha != default value')
        self.assertEqual(model.sampling_parameters['a_gamma'], 1, msg='a_gamma != default value')
        self.assertEqual(model.sampling_parameters['b_gamma'], 0.001, msg='b_gamma != default value')
        self.assertEqual(model.sampling_parameters['c'], 100, msg='c != default value')
        self.assertEqual(model.sampling_parameters['d'], 1, msg='d != default value')
        self.assertEqual(model.training_parameters['iterations'], 1000, msg='iterations != default value')
        self.assertEqual(model.training_parameters['save_every'], 100, msg='save_every != default value')
        self.assertEqual(model.training_parameters['print_every'], 100, msg='print_every != default value')
        self.assertEqual(model.training_parameters['burn_in'], 100, msg='burn_in != default value')
        self.assertEqual(model.DP_parameters['gamma'], 1 / 0.001, msg='gamma != default value')
        self.assertEqual(model.DP_parameters['alpha_p_kappa'], 1 / 0.001,
                         msg='alpha_p_kappa != default value')
        self.assertEqual(model.DP_parameters['rho'], 100 / 101, msg='rho != default value')

        try:
            np.testing.assert_array_equal(model.sampling_parameters['K'],
                                          np.linalg.inv(np.diag(0.5 * np.ones(self.D * self.order))),
                                          err_msg='K != default value')
        except AssertionError:
            self.assertFalse(True, msg='K != default value')
        try:
            np.testing.assert_array_equal(model.sampling_parameters['M'],
                                          np.zeros((self.D, self.D * self.order)),
                                          err_msg='M != default value')
        except AssertionError:
            self.assertFalse(True, msg='M != default value')
        try:
            np.testing.assert_array_equal(model.sampling_parameters['S_0'],
                                          np.eye(self.D),
                                          err_msg='S_0 != default value')
        except AssertionError:
            self.assertFalse(True, msg='S_0 != default value')
        try:
            np.testing.assert_array_equal(model.theta['A'],
                                          np.zeros((self.D, self.D * self.order, self.L)),
                                          err_msg='A != default value')
        except AssertionError:
            self.assertFalse(True, msg='A != default value')
        try:
            np.testing.assert_array_equal(model.theta['Sigma'],
                                          np.zeros((self.D, self.D, self.L)),
                                          err_msg='Sigma != default value')
        except AssertionError:
            self.assertFalse(True, msg='Sigma != default value')

    def test_set(self):
        """

        :return:
        """
        sampling_params = {'K': np.linalg.inv(np.diag(np.ones(self.D * self.order))),
                           'M': np.zeros((self.D, self.D * self.order)), 'S_0': 0.5 * np.eye(self.D),
                           'n_0': self.D + 2, 'a_gamma': 1, 'b_gamma': 1, 'a_alpha': 1,
                           'b_alpha': 1, 'c': 1, 'd': 1}
        training_params = {'iterations': 100, 'save_every': 10, 'print_every': 10, 'burn_in': 10}
        model = HDPVar(order=self.order, D=self.D, L=self.L)
        model.set_sampling_parameters(sampling_params)
        model.set_training_parameters(training_params)
        try:
            np.testing.assert_array_equal(model.sampling_parameters['K'], sampling_params['K'])
        except AssertionError:
            self.assertFalse(True, msg='K != set value')
        try:
            np.testing.assert_array_equal(model.sampling_parameters['M'], sampling_params['M'])
        except AssertionError:
            self.assertFalse(True, msg='M != set value')
        try:
            np.testing.assert_array_equal(model.sampling_parameters['S_0'], sampling_params['S_0'])
        except AssertionError:
            self.assertFalse(True, msg='S_0 != set value')
        self.assertEqual(model.sampling_parameters['n_0'], sampling_params['n_0'], msg='n_0 != set value')
        self.assertEqual(model.sampling_parameters['a_gamma'], sampling_params['a_gamma'], msg='a_gamma != set value')
        self.assertEqual(model.sampling_parameters['b_gamma'], sampling_params['b_gamma'], msg='b_gamma != set value')
        self.assertEqual(model.sampling_parameters['a_alpha'], sampling_params['a_alpha'], msg='a_alpha != set value')
        self.assertEqual(model.sampling_parameters['b_alpha'], sampling_params['b_alpha'], msg='b_alpha != set value')
        self.assertEqual(model.sampling_parameters['c'], sampling_params['c'], msg='c != set value')
        self.assertEqual(model.sampling_parameters['d'], sampling_params['d'], msg='c != set value')
        self.assertEqual(model.training_parameters['iterations'], training_params['iterations'],
                         msg='iterations != set value')
        self.assertEqual(model.training_parameters['save_every'], training_params['save_every'],
                         msg='save_every != set value')
        self.assertEqual(model.training_parameters['print_every'], training_params['print_every'],
                         msg='print_every != set value')
        self.assertEqual(model.training_parameters['burn_in'], training_params['burn_in'],
                         msg='burn_in != set value')
        self.assertEqual(model.DP_parameters['gamma'], 1, msg='gamma != set value')
        self.assertEqual(model.DP_parameters['alpha_p_kappa'], 1, msg='alpha_p_kappa != set value')
        self.assertEqual(model.DP_parameters['rho'], 0.5, msg='rho != set value')

    def test_distributions(self):
        """

        :return:
        """
        model = HDPVar(order=self.order, D=self.D, L=self.L)
        model.sample_distributions(seed=42)
        try:
            np.testing.assert_array_almost_equal(model.distributions['transition_probabilities'],
                                                 np.array([[0.99474837, 0.00525163], [0.0044258, 0.9955742]]))
        except AssertionError:
            self.assertFalse(True, msg='transition_probabilities != default value')
        try:
            np.testing.assert_array_almost_equal(model.distributions['initial_probabilities'],
                                                 np.array([0.45174329, 0.54825671]))
        except AssertionError:
            self.assertFalse(True, msg='initial_probabilities != default value')
        try:
            np.testing.assert_array_almost_equal(model.distributions['beta'],
                                                 np.array([0.49649512, 0.50350488]))
        except AssertionError:
            self.assertFalse(True, msg='beta != default value')

    def test_theta(self):
        """

        :return:
        """
        model = HDPVar(order=self.order, D=self.D, L=self.L)
        model.sample_init_theta(seed=42)
        try:
            np.testing.assert_array_almost_equal(model.theta['A'],
                                                 np.array([[[0.2830998, 0.2830998],
                                                            [0.36914692, 0.36914692],
                                                            [-0.13345457, -0.13345457],
                                                            [0.90006463, 0.90006463]],
                                                           [[-0.38974281, -0.38974281],
                                                            [0.99489065, 0.99489065],
                                                            [-0.08040436, -0.08040436],
                                                            [-0.17398075, -0.17398075]]]))
        except AssertionError:
            self.assertFalse(True, msg='A != default value')
        try:
            np.testing.assert_array_almost_equal(model.theta['Sigma'],
                                                 np.array([[[0.4593895, 0.4593895],
                                                            [-0.43445108, - 0.43445108]],
                                                           [[-0.43445108, - 0.43445108],
                                                            [1.51214062, 1.51214062]]]))
        except AssertionError:
            self.assertFalse(True, msg='Sigma != default value')


if __name__ == '__main__':
    unittest.main()
