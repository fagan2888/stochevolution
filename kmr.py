from __future__ import division
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
import quantecon as qe


def kmr_markov_matrix_simultaneous(p, N, epsilon):
    P = np.empty((N+1, N+1))
    for n in range(N+1):
        P[n, :] = \
            (n/N < p) * binom.pmf(range(N+1), N, epsilon/2) + \
            (n/N == p) * binom.pmf(range(N+1), N, 1/2) + \
            (n/N > p) * binom.pmf(range(N+1), N, 1-epsilon/2)
    return P


def kmr_markov_matrix_sequential(p, N, epsilon):
    P = np.zeros((N+1, N+1))
    P[0, 0], P[0, 1] = 1 - epsilon * (1/2), epsilon * (1/2)
    for n in range(1, N):
        P[n, n-1] = (n/N) * (
            epsilon * (1/2) +
            (1 - epsilon) * (((n-1)/(N-1) < p) + ((n-1)/(N-1) == p) * (1/2))
        )
        P[n, n+1] = ((N-n)/N) * (
            epsilon * (1/2) +
            (1 - epsilon) * ((n/(N-1) > p) + (n/(N-1) == p) * (1/2))
        )
        P[n, n] = 1 - P[n, n-1] - P[n, n+1]
    P[N, N-1], P[N, N] = epsilon * (1/2), 1 - epsilon * (1/2)
    return P


class KMR2x2(object):

    def __init__(self, p, N, epsilon, revision='simultaneous'):

        self.p, self.N = p, N
        self._epsilon = epsilon
        self._mc = None

        if revision == 'simultaneous':
            self.kmr_markov_matrix = kmr_markov_matrix_simultaneous
        elif revision == 'sequential':
            self.kmr_markov_matrix = kmr_markov_matrix_sequential
        else:
            raise ValueError

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value
        self._mc = None

    @property
    def mc(self):
        if self._mc is None:
            P = self.kmr_markov_matrix(self.p, self.N, self.epsilon)
            self._mc = qe.MarkovChain(P)
        return self._mc

    def simulate(self, ts_length, init=None, num_reps=None):
        """
        Simulate the dynamics.

        Parameters
        ----------
        ts_length : scalar(int)
            Length of each simulation.

        init : scalar(int) or array_like(int, ndim=1),
               optional(default=None)
            Initial state(s). If None, the initial state is randomly
            drawn.

        num_reps : scalar(int), optional(default=None)
            Number of simulations. Relevant only when init is a scalar
            or None.

        Returns
        -------
        ndarray(int, ndim=1 or 2)
            Array containing the sample path(s), of shape (ts_length,)
            if init is a scalar (integer) or None and num_reps is None;
            of shape (k, ts_length) otherwise, where k = len(init) if
            init is an array_like, otherwise k = num_reps.

        """
        return self.mc.simulate(ts_length, init, num_reps)

    def plot_sample_path(self, x=None, ts_length=10**4, init=None,
                         ax=None, show=True):
        if x is None:
            x = self.simulate(ts_length, init)

        if show:
            fig, ax = plt.subplots()
        ax.plot(x, alpha=0.5)
        ax.set_ylim(0, self.N)
        ax.set_title(r'Sample path: $\varepsilon = {0}$'.format(self._epsilon))
        ax.set_xlabel('Time')
        ax.set_ylabel('State')
        if show:
            plt.show()

    def plot_emprical_dist(self, x=None, ts_length=10**4, init=None,
                           ax=None, show=True):
        if x is None:
            x = self.simulate(ts_length, init)

        if show:
            fig, ax = plt.subplots()
        hist, bins = np.histogram(x, self.N+1)
        ax.bar(range(self.N+1), hist, align='center')
        ax.set_title(r'Emprical distribution: $\varepsilon = {0}$'
                     .format(self.epsilon))
        ax.set_xlim(-0.5, self.N+0.5)
        ax.set_xlabel('State')
        ax.set_ylabel('Frequency')
        if show:
            plt.show()

    @property
    def stationary_dist(self):
        """
        Return an array containing the stationary distribution

        """
        return self.mc.stationary_distributions[0]

    def plot_stationary_dist(self, ax=None, show=True):
        if show:
            fig, ax = plt.subplots()
        ax.bar(range(self.N+1), self.stationary_dist, align='center')
        ax.set_xlim(-0.5, self.N+0.5)
        ax.set_ylim(0, 1)
        ax.set_title(r'Stationary distribution: $\varepsilon = {0}$'
                     .format(self.epsilon))
        ax.set_xlabel('State')
        ax.set_ylabel('Probability')
        if show:
            plt.show()
