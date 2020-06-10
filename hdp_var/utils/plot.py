"""

"""
import matplotlib.pyplot as plt


def plot_likelihood(model):
    """

    :param model:
    :return:
    """
    iterations = list(range(model.training_parameters['burn_in'],
                            model.training_parameters['iterations'], model.training_parameters['sample_every']))
    if model.training_parameters['iterations'] not in iterations:
        iterations.append(model.training_parameters['iterations'])
    fig, ax = plt.subplots()
    ax.plot(iterations, model.total_log_likelihoods)
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def plot(data_1, data_2):
    """

    :param data_1:
    :param data_2:
    :return:
    """
    fig, ax = plt.subplots()
    ax.plot(data_1, label='data_1')
    ax.plot(data_2, label='data_2')
    plt.xlabel('Time')
    plt.ylabel('degrees')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def plot_1(data_1):
    """

    :param data_1:
    :param data_2:
    :return:
    """
    fig, ax = plt.subplots()
    ax.plot(data_1)
    plt.xlabel('Time')
    plt.ylabel('degrees')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
