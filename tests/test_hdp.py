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

    N = int(1e5)
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
    mcmc = HDPNormalMixture(data, ncomp=3, gpu=[0,1,2,3], parallel=True, verbose=100)
    mcmc.sample(1000, nburn=1000, tune_interval=100)
    imcmc = HDPNormalMixture(mcmc, verbose=100)
    imcmc.sample(2000, nburn=0, ident=True)
    print imcmc.mu[-1]
    print imcmc.weights[-1]
    print imcmc.beta[-1]


    

