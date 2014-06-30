from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mc_tools import mc_compute_stationary, mc_sample_path

p = 1/3
N = 10
# epsilons = [0.1, 0.05, 0.01, 0.001]
epsilon = 0.01
T = 500000

def KMR_2x2_P_sequential(N, p, epsilon):
    P = np.zeros((N+1, N+1), dtype=float)
    P[0, 0], P[0, 1] = 1 - epsilon * (1/2), epsilon * (1/2)
    for n in range(1, N):
        P[n, n-1] = (n/N) * \
                    (epsilon * (1/2) +
                     (1 - epsilon) * ((n/(N-1) < p) + (n/(N-1) == p) * (1/2))
                     )
        P[n, n+1] = ((N-n)/N) * \
                    (epsilon * (1/2) +
                     (1 - epsilon) * ((n/(N-1) > p) + (n/(N-1) == p) * (1/2))
                     )
        P[n, n] = 1 - P[n, n-1] - P[n, n+1]
    P[N, N-1], P[N, N] = epsilon * (1/2), 1 - epsilon * (1/2)
    return P


"""
mus = []
for epsilon in epsilons:
    mu = mc_compute_stationary(KMR_2x2_P_sequential(N, p, epsilon))
    mus.append(mu)

np.set_printoptions(precision=8, suppress=True)
for epsilon, mu in zip(epsilons, mus):
    print('epsilon = {0}: {1}'.format(epsilon, mu))
"""

P = KMR_2x2_P_sequential(N, p, epsilon)

x0 = 0
X = mc_sample_path(P, x0, T)

fig, ax = plt.subplots()
ax.plot(X)
ax.set_ylim(0, N)
plt.show()
