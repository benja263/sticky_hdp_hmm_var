import attr


@attr.s
class SamplingParams:
    """
    :param K:
    :param M:
    :param S_0:
    :param n_0:
    :param a_gamma:
    :param b_gamma:
    :param a_alpha:
    :param b_alpha:
    :param c:
    :param d:
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
    :param iterations:
    :param sample_every:
    :param print_every:
    :param burn_in:
    """
    iterations = attr.ib(default=None)
    sample_every = attr.ib(default=None)
    print_every = attr.ib(default=None)
    burn_in = attr.ib(default=None)

