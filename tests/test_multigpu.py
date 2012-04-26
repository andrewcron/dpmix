"""
Created on April 5, 2012

@authoer: Andrew Cron
"""

import sys
import time
sys.path.append("../src")

from test_util import *

import numpy as np
import multigpu

if __name__ == '__main__':

    N = int(1e5)
    K = 2
    J = 2
    ncomps = 4
    gpus = [0,1,2,3,4]
    true_labels, data = generate_data(n=N, k=K, ncomps=3)
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
    #import pdb; pdb.set_trace()
    workers = multigpu.init_GPUWorkers(data, gpus)
    multigpu.start_GPUWorkers(workers)
    starttime = time.time()
    for i in xrange(2000):
        if i % 50 == 0:
            print i
        ll, ct, xbar, dens = multigpu.get_expected_labels_GPU(workers, w, mu, Sigma)
        labels = multigpu.get_labelsGPU(workers, w, mu, Sigma, True)
        #import pdb; pdb.set_trace()


    multigpu.kill_GPUWorkers(workers)

    print "DONE! it took " + str(time.time() - starttime)

    
