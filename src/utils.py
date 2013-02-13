'''
Created on Mar 15, 2012

@author: Androw Cron
@author: Jacob Frelinger
'''

import numpy as np
import scipy.linalg as LA
import scipy.stats as stats

import cython

def select_gpu(devNum):
    """
    sets the devNum device as the active GPU if it's not already
    active. Assumes that GPU capability has been checked.
    """
    import pycuda.autoinit as cudainit
    import pycuda.driver as drv
    context = cudainit.context
    curDev = cudainit.device
    newDev = drv.Device(devNum)
    if curDev != newDev:
        context.pop()
        cudainit.context = newDev.make_context()
        cudainit.device = newDev


# @cython.compile
# def _get_cost(x,y,C):
#     n = len(x)
#     for i in range(n):
#         C[x[i], y[i]] -= 1 


def mvn_weighted_logged(data, means, covs, weights):
    n, p = data.shape
    k = len(weights)
    densities = np.zeros((n, k))
    const = 0.5 * p * np.log(2*np.pi)

    for i in range(k):
        if covs[i,:,:].shape == (1,1):
            chol = np.sqrt(covs[i,:,:])
            if chol == 0:
                chol = np.array([1e-5])
            diff = (data - means[i,:]) * 1.0/chol
        else:
            chol = LA.cholesky(covs[i,:,:])
            diff = np.dot(data - means[i,:], LA.inv(chol))
        try:
            densities[:,i] = np.log(weights[i]) - const - np.log(np.diag(chol)).sum() - 0.5*(diff**2).sum(1)
        except ValueError:
            print chol
            print np.diag(chol)
            raise ValueError

    return densities
    

def sample_discrete(densities, logged=True):
    # this will sample the discrete densities
    # if they are logged, they will be exponentiated IN PLACE
    
    
    n, p = densities.shape
    
    #if we're logged exponentiate.
    if logged:
        densities = np.exp((densities.T - densities.max(1)).T)
    norm = densities.sum(1)
    
    densities = (densities.T / norm).T
    
    # calculate a cumlative probability over all the densities
    # draw a random [0,1] for each and measure across to find
    # where it swaps from True to False
    # uses True == 1 to quickly find the cross over point point
    # since K - #of True == index of first True
    
    csx = np.cumsum(densities, axis=1)
    r = np.random.random((n,1))

    y = r < csx
    labels = len(y[0]) - np.sum(y, 1)

    return labels
    
def stick_break_proc(beta_a, beta_b, size=None):
    """
    Kernel stick breaking procedure for truncated Dirichlet Process

    Parameters
    ----------
    beta_a : scalar or array-like
    beta_b : scalar or array-like
    size : int, default None
        If array-like a, b, leave as None

    Notes
    -----


    Returns
    -------
    (stick_weights, mixture_weights) : (1d ndarray, 1d ndarray)
    """
    if not np.isscalar(beta_a):
        size = len(beta_a)
        assert(size == len(beta_b))
    else:
        assert(size is not None)

    # check for zeros ... 
    beta_a[beta_a<1e-10] = 1e-10
    beta_b[beta_b<1e-10] = 1e-10

    dist = stats.beta(beta_a, beta_b)
    V = stick_weights = dist.rvs(size)
    #check for bad values and deal with adhoc
    V[V<1e-10] = 1e-10; V[V>(1-1e-10)] = 1-1e-10
    nanmask = np.isnan(V)

    if np.sum(nanmask)>0:
        value = beta_a / (beta_a + beta_b)
        V[nanmask] = value[nanmask]
        
        
    pi = mixture_weights = np.empty(len(V) + 1)

    pi[0] = V[0]
    prod = (1 - V[0])
    for k in xrange(1, len(V)):
        pi[k] = prod * V[k]
        prod *= 1 - V[k]
    pi[-1] = prod

    '''
    v_cumprod = (1 - V).cumprod()
    pi[1:-1] = V[1:] * v_cumprod[:-1]
    pi[-1] = v_cumprod[-1]
    '''
    return stick_weights, mixture_weights

def _get_mask(labels, ncomp):
    return np.equal.outer(np.arange(ncomp), labels)

# stick breaking func
#@cython.compile
def break_sticks(V):
    n = len(V)
    pi = np.empty(n+1)
    pi[0] = V[0]
    prod = (1-V[0])
    for k in xrange(1, n):
        pi[k] = prod * V[k]
        prod *= 1 - V[k]
    pi[-1] = prod
    return pi


#### Task Classes ####
class MCMC_Task(object):
    def __init__(self, ncomp, relabel=False, dataind=0, gid = 0):
        self.ncomp = ncomp
        self.relabel = relabel
        self.dataind = dataind
        self.gid = gid

class BEM_Task(object):

    def __init__(self, ncomp, dataind=0, gid=0):
        self.ncomp = ncomp
        self.dataind = dataind
        self.gid = gid

class Init_Task(object):

    def __init__(self, dim1, dim2, dev_num):
        #self.data = data
        self.nobs, self.ndim = dim1, dim2
        self.dev_num = dev_num
