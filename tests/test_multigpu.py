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

import gpustats as gs

if __name__ == '__main__':

    N = int(1e5)
    K = 2
    J = 2
    ncomps = 4
    gpus = [2,3]
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

    workers = multigpu.init_GPUWorkers(data, gpus)

    starttime = time.time()
    for i in xrange(1000):
        if i % 100 == 0:
            print i
            #import pdb; pdb.set_trace()
        ll, ct, xbar, dens = multigpu.get_expected_labels_GPU(workers, w, mu, Sigma)
        labels = multigpu.get_labelsGPU(workers, w, mu, Sigma, True)

    ## make sure host GPU is ok ... 
    from pycuda.gpuarray import to_gpu 
    from pycuda.gpuarray import sum as gsum
    test = to_gpu(np.ones(100000, dtype=np.int32))
    print gsum(test)

    multigpu.kill_GPUWorkers(workers)

    print "DONE! it took " + str(time.time() - starttime)

    
