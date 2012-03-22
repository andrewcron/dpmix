'''
Creaded on Mar 21, 2012

@author: Andrew Cron
'''

import sys
sys.path.append('../src')

from test_dpmix import *

import numpy as np

from hdp import HDPNormalMixture
import pylab

if __name__ == '__main__':

    N = int(1e4)
    K = 2
    J = 4
    ncomps = 3
    true_labels, data = generate_data(n=N, k=K, ncomps=ncomps)
    data = data - data.mean(0)
    data = data/data.std(0)
    #shuffle the data ... 
    ind = np.arange(N); np.random.shuffle(ind);
    all_data = data[ind].copy()
    data = [ all_data[(N/J*i):(N/J*(i+1))].copy() for i in range(J) ]
    mcmc = HDPNormalMixture(data, ncomp=3, gpu=False)
    mcmc.sample(2000, nburn=1000, tune_interval=100)
    import pdb
    pdb.set_trace()
    

