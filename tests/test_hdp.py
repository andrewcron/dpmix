"""
Creaded on Mar 21, 2012

@author: Andrew Cron
"""

import sys
sys.path.insert(0, './src')

from test_help import *

import numpy as np

from hdp import HDPNormalMixture


if __name__ == '__main__':
    np.random.seed(123)

    J = 3
    N = int(1e5)
    K = 2
    ncomps = 3
    true_labels, data = generate_data(n=N, k=K, ncomps=ncomps)
    data = data - data.mean(0)
    data = data/data.std(0)

    # shuffle the data...
    ind = np.arange(N)
    np.random.shuffle(ind)
    all_data = data[ind].copy()
    data = [all_data[(N/J*i):(N/J*(i+1))].copy() for i in range(J)]

    mcmc = HDPNormalMixture(
        data, ncomp=10, parallel=False, verbose=1, gpu=True
    )
    mcmc.sample(niter=2, nburn=8, tune_interval=100)

    print mcmc.mu[-1][-1]
