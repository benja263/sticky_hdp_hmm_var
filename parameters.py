import attr


@attr.s
class SamplingParams:
    """
    :param K: Matrix Normal prior precision matrix
    :param M: Matrix Normal prior mean
    :param S_0: Inverse Wishart prior scale matrix
    :param n_0: Inverse Wishart prior degrees of freedom
    :param a_gamma: a parameter for the gamma DP parameter
    :param b_gamma: b parameter for the gamma DP parameter
    :param a_alpha: a parameter for the alpha DP parameter
    :param b_alpha: b parameter for the alpha DP parameter
    :param c: c parameter for the Beta prior for rho
    :param d: d parameter for the Beta prior for rho
    """
    K = attr.ib(default=None)
    M = attr.ib(default=None)
    S_0 = attr.ib(default=None)
    n_0 = attr.ib(default=None)
    a_gamma = attr.ib(default=None)
    b_gamma = attr.ib(default=None)
    a_alpha = attr.ib(default=None)
    b_alpha = attr.ib(default=None)
    c = attr.ib(default=None)
    d = attr.ib(default=None)


@attr.s
class TrainingParams:
    """
    :param int iterations: number of training iterations
    :param int sample_every: number of iterations between sampling
    :param int print_every: number of iterations between printing
    :param int burn_in: number of burn_in iterations
    """
    iterations = attr.ib(default=None)
    sample_every = attr.ib(default=None)
    print_every = attr.ib(default=None)
    burn_in = attr.ib(default=None)

