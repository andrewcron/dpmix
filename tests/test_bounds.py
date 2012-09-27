import numpy.random as npr
from dpmix import HDPNormalMixture


if __name__ == '__main__':
    nclust = 256
    niter = 10
    burnin = 10
    device = 2
    max_events = 50000
    num_files = 10

    seed = 9
    #npr.seed(seed)
    for it in range(10, 20):
        xs = []
        for i in range(num_files):
            print i,
            xs.append(npr.uniform(-5,5,(max_events, 5)))
        print
        mcmc = HDPNormalMixture(xs, ncomp=nclust, gpu=device, parallel=True, verbose=2)
        mcmc.sample(burnin, nburn=0, tune_interval=5)
        imcmc = HDPNormalMixture(mcmc, verbose=2)
        imcmc.sample(niter, nburn=0, ident=True)

        del mcmc
        del imcmc
