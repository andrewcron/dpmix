'''
Creaded on Mar 21, 2012

@author: Andrew Cron
'''


import sys
sys.path.insert(0,'../src')

from test_help import *

import numpy as np

from hdp import HDPNormalMixture
#from dpmix import HDPNormalMixture

#import gpustats as gs

if __name__ == '__main__':

    N = int(1e6)
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

    mcmc = HDPNormalMixture(data, ncomp=4, gpu=[1,3], parallel=True, verbose=100)
    mcmc.sample(200, nburn=500, tune_interval=100)
    imcmc = HDPNormalMixture(mcmc, verbose=100)
    imcmc.sample(200, nburn=0, ident=True)
    print imcmc.mu[-1]
    print imcmc.weights[-1]
    print imcmc.beta[-1]



    

