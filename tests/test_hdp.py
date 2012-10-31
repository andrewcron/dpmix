'''
Creaded on Mar 21, 2012

@author: Andrew Cron
'''


import sys
sys.path.insert(0,'./src')

from test_help import *

import numpy as np

from hdp import HDPNormalMixture
#from dpmix import HDPNormalMixture

#import gpustats as gs

if __name__ == '__main__':

    J = 20
    N = int(1e7)
    K = 2
    ncomps = 3
    true_labels, data = generate_data(n=N, k=K, ncomps=ncomps)
    data = data - data.mean(0)
    data = data/data.std(0)
    #shuffle the data ... 
    ind = np.arange(N); np.random.shuffle(ind);
    all_data = data[ind].copy()
    data = [ all_data[(N/J*i):(N/J*(i+1))].copy() for i in range(J) ]

    #mcmc = HDPNormalMixture(data, ncomp=3, gpu=[0,1,2], parallel=True, verbose=100)
    mcmc = HDPNormalMixture(data, ncomp=100, parallel=True, verbose=1,gpu=[0,1,2,3,4])
    mcmc.sample(2, nburn=1, tune_interval=50)
    #import pdb; pdb.set_trace()
    #imcmc = HDPNormalMixture(mcmc, verbose=100)
    #imcmc.sample(100, nburn=0, ident=True)
    #print imcmc.mu[-1]
    #print imcmc.weights[-1]
    #print imcmc.beta[-1]



    

