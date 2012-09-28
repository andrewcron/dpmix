import numpy as np
from time import time
from dpmix import HDPNormalMixture

# load data
files = ['AMJ_5L_CMV pp65.npy',
         'AMJ_5L_Costim.npy', 
         'AMJ_5L_SEB.npy']
data = [np.load(fl) for fl in files]

#shift and scale (all data!!)
all_data = data[0]
for i in range(1,3):
    all_data = np.r_[all_data, data[i]]

dmean = all_data.mean(0)
dstd = all_data.std(0)

for d in data:
    d -= dmean
    d /= dstd


## run some benchmarks!
if __name__ == '__main__':

    t1 = time()
    mcmc = HDPNormalMixture(data, ncomp=256, gpu=[0,1,2], 
                            parallel=True, verbose=100)
    mcmc.sample(25, nburn=25, tune_interval=50)
    imcmc = HDPNormalMixture(mcmc, verbose=100)
    imcmc.sample(250, nburn=0, ident=True)
    t1 = time() - t1
    print 'ALL GPU: ' + str(t1)

    t2 = time()
    mcmc = HDPNormalMixture(data, ncomp=256, gpu=[0], 
                            parallel=False, verbose=100)
    mcmc.sample(250, nburn=500, tune_interval=50)
    imcmc = HDPNormalMixture(mcmc, verbose=100)
    imcmc.sample(250, nburn=0, ident=True)
    t2 = time() - t2
    print 'One GPU: ' + str(t2)

    t4 = time()
    mcmc = HDPNormalMixture(data, ncomp=256, gpu=False, 
                            parallel=False, verbose=10)
    mcmc.sample(25, nburn=50, tune_interval=50)
    imcmc = HDPNormalMixture(mcmc, verbose=10)
    imcmc.sample(25, nburn=0, ident=True)
    t4 = time() - t4
    print 'One CPU: ' + str(t4)

    t3 = time()
    mcmc = HDPNormalMixture(data, ncomp=256, gpu=False, 
                            parallel=True, verbose=10)
    mcmc.sample(25, nburn=50, tune_interval=50)
    imcmc = HDPNormalMixture(mcmc, verbose=10)
    imcmc.sample(25, nburn=0, ident=True)
    t3 = time() - t3
    print 'ALL CPU: ' + str(t3)
    



