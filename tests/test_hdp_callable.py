import sys
sys.path.insert(0, './src')

from test_help import *

import numpy as np

from hdp import HDPNormalMixture


class Callable(object):
    def __init__(self, nburn, niter):
        self.nburn = nburn
        self.niter = niter

    def __call__(self, iteration):
        print "%.2f%%" % self.calc_percent_complete(iteration)

    def calc_percent_complete(self, iteration):
        # the burn-in iterations are negative
        # while the n-iterations are positive
        return (iteration + self.nburn + 1) * 100 / (self.nburn + self.niter)


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
        data, ncomp=10, parallel=False, verbose=0, gpu=True
    )

    nburn = 20
    niter = 2
    iter_tracker = Callable(nburn, niter)

    mcmc.sample(niter=niter, nburn=nburn, tune_interval=100, callback=iter_tracker)

    print mcmc.mu[-1][-1]
