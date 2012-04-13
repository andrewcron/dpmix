"""
Created on April 5, 2012

@authoer: Andrew Cron
"""

import sys
sys.path.append("../src")

from test_dpmix import *

import numpy as np
import multigpu

from dpmix import DPNormalMixture

if __name__ == '__main__':

    N = 3*int(1e4)
    K = 2
    J = 2
    ncomps = 3
    gpus = [2,3,4]
    true_labels, data = generate_data(n=N, k=K, ncomps=ncomps)
    data = data - data.mean(0)
    data = data/data.std(0)
    #shuffle the data ... 
    ind = np.arange(N); np.random.shuffle(ind);
    all_data = data[ind].copy()

    w = np.ones(ncomps)
    mu = np.zeros((ncomps, J))
    Sigma = np.zeros((ncomps, J, J))
    for i in range(ncomps):
        Sigma[i] = np.identity(J)
    import pdb; pdb.set_trace()
    workers = multigpu.init_GPUWorkers(data, w, mu, Sigma, gpus)
    labels = multigpu.get_labels(workers, w, mu, Sigma)
    multigpu.kill_workers(workers)
    print "DONE!"
    
