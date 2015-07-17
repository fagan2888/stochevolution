from __future__ import division
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
from mc_tools import mc_compute_stationary, mc_sample_path


def KMR_2x2_P_simultaneous(N, p, epsilon):
    P = np.empty((N+1, N+1), dtype=float)
    for n in range(N+1):
        P[n, :] = \
            (n/N < p) * binom.pmf(range(N+1), N, epsilon/2) + \
            (n/N == p) * binom.pmf(range(N+1), N, 1/2) + \
            (n/N > p) * binom.pmf(range(N+1), N, 1-epsilon/2)
    return P


def KMR_2x2_P_sequential(N, p, epsilon):
    P = np.zeros((N+1, N+1), dtype=float)
    P[0, 0], P[0, 1] = 1 - epsilon * (1/2), epsilon * (1/2)
    for n in range(1, N):
        P[n, n-1] = \
            (n/N) * (epsilon * (1/2) +
                     (1 - epsilon) * (((n-1)/(N-1) < p) + ((n-1)/(N-1) == p) * (1/2))
                     )
        P[n, n+1] = \
            ((N-n)/N) * (epsilon * (1/2) +
                         (1 - epsilon) * ((n/(N-1) > p) + (n/(N-1) == p) * (1/2))
                         )
        P[n, n] = 1 - P[n, n-1] - P[n, n+1]
    P[N, N-1], P[N, N] = epsilon * (1/2), 1 - epsilon * (1/2)
    return P


class KMR_2x2:

    def __init__(self, N, p, epsilon, move='simultaneous'):
        self._epsilon = epsilon
        self.N, self.p, self.move = N, p, move
        self.set_P()

    def get_epsilon(self):
        return self._epsilon

    def set_epsilon(self, new_value):
        self._epsilon = new_value
        self.set_P()

    epsilon = property(get_epsilon, set_epsilon)

    def set_P(self):
        if self.move == 'sequential':
            self.P = KMR_2x2_P_sequential(self.N, self.p, self._epsilon)
        else:
            self.P = KMR_2x2_P_simultaneous(self.N, self.p, self._epsilon)

    def simulate(self, T=100000, x0=0):
        """
        Generates a NumPy array containing a sample path of length T
        with initial state x0 = 0
        """
        self.s = mc_sample_path(self.P, x0, T)

    def get_sample_path(self):
        return self.s

    def plot_sample_path(self, ax=None, show=True):
        if show:
            fig, ax = plt.subplots()
        ax.plot(self.s, alpha=0.5)
        ax.set_ylim(0, self.N)
        ax.set_title(r'Sample path: $\varepsilon = {0}$'.format(self._epsilon))
        ax.set_xlabel('time')
        ax.set_ylabel('state space')
        if show:
            plt.show()

    def plot_emprical_dist(self, ax=None, show=True):
        if show:
            fig, ax = plt.subplots()
        hist, bins = np.histogram(self.s, self.N+1)
        ax.bar(range(self.N+1), hist, align='center')
        ax.set_title(r'Emprical distribution: $\varepsilon = {0}$'.format(self._epsilon))
        ax.set_xlim(-0.5, self.N+0.5)
        ax.set_xlabel('state space')
        ax.set_ylabel('frequency')
        if show:
            plt.show()

    def compute_stationary_dist(self):
        """
        Generates a NumPy array containing the stationary distribution
        """
        self.mu = mc_compute_stationary(self.P)

    def get_stationary_dist(self):
        return self.mu

    def plot_stationary_dist(self, ax=None, show=True):
        if show:
            fig, ax = plt.subplots()
        ax.bar(range(self.N+1), self.mu, align='center')
        ax.set_xlim(-0.5, self.N+0.5)
        ax.set_ylim(0, 1)
        ax.set_title(r'Stationary distribution: $\varepsilon = {0}$'.format(self._epsilon))
        ax.set_xlabel('state space')
        ax.set_ylabel('probability')
        if show:
            plt.show()


if __name__ == '__main__':
    p = 1/3
    N = 10
    epsilon = 0.03
    T = 300000

    kmr = KMR_2x2(N, p, epsilon)
    kmr.simulate(T)
    kmr.plot_sample_path()
