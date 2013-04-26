'''
Created on Mar 15, 2012

@author: Andrew Cron
@author: Jacob Frelinger
'''
import sys
#sys.path.insert(0, '../build/lib.linux-x86_64-2.7/')
sys.path.insert(0, "./src")

import numpy as np
import numpy.random as npr
import pymc as pm
#
from dpmix import DPNormalMixture
from BEM import BEM_DPNormalMixture
#from dpmix import DPNormalMixture, BEM_DPNormalMixture

#import gpustats as gs

from test_help import *

def plot_2d_mixture(data, labels):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    colors = 'bgr'
    for j in np.unique(labels):
        x, y = data[labels == j].T
        plt.plot(x, y, '%s.' % colors[j], ms=2)

if __name__ == '__main__':
    from datetime import datetime
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("--gpu", default=False)
    parser.add_option("--verbose", default=False)
    parser.add_option("--parallel", default=False, action="store_true")
    (options, args) = parser.parse_args()

    if type(options.gpu) is bool:
        use_gpu = options.gpu
    elif options.gpu == 'ALL':
        use_gpu = [0,1,2]
    elif options.gpu == 'MPI':
        use_gpu = {'lilo': 0, 'stitch' : 0 }
    else:
        use_gpu = int(options.gpu)
    verbosity = int(options.verbose)

    N = int(1e5) # n data points per component
    K = 2 # ndim
    ncomps = 3 # n mixture components
    npr.seed(datetime.now().microsecond)
    true_labels, data = generate_data(n=N, k=K, ncomps=ncomps)
    data = data - data.mean(0)
    data = data/data.std(0)

    #import pdb
    #pdb.set_trace()
    print "use_gpu=" + str(use_gpu)
    mcmc = DPNormalMixture(data, ncomp=3, gpu=use_gpu, verbose=verbosity, 
                           parallel=options.parallel)#, mu0=mu0)
    mcmc.sample(100,nburn=1000)
    print mcmc.mu[-1]
    print mcmc.Sigma[-1]

    bem = BEM_DPNormalMixture(mcmc, verbose=verbosity)
    bem.optimize(maxiter=5)
    print bem.mu
    
    ident_mcmc = DPNormalMixture(bem, verbose=verbosity)
    ident_mcmc.sample(100, nburn=0, ident=False)
    print ident_mcmc.weights[-1]
    print ident_mcmc.mu[-1]



