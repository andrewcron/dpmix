'''
Created on Mar 15, 2012

@author: Andrew Cron
'''
import numpy as np
import scipy.linalg as LA
import scipy.stats as stats

from utils import mvn_weighted_logged
from dpmix import DPNormalMixture

#from multicpu import BEMSigmaUpdate
# check for gpustats compatability
try:
    import pycuda
    import pycuda.driver
    try:
        from multigpu import init_GPUWorkers, get_expected_labels_GPU, kill_GPUWorkers
        
        _has_gpu = True
    except (ImportError, pycuda.driver.RuntimeError):
        _has_gpu=False
except ImportError:
    _has_gpu = False

class BEM_DPNormalMixture(DPNormalMixture):
    """
    BEM algorithm for finding the posterior mode of the
    Truncated Dirichlet Process Mixture of Models 

    Parameters
    ----------
    data : ndarray (nobs x ndim)  or (BEM_)DPNormalMixture class
    ncomp : int
        Number of mixture components

    Notes
    -----
    y ~ \sum_{j=1}^J \pi_j {\cal N}(\mu_j, \Sigma_j)
    \alpha ~ Ga(e, f)
    \Sigma_j ~ IW(nu0 + 2, nu0 * \Phi_j)

    Citation
    --------

    M. Suchard, Q. Wang, C. Chan, J. Frelinger, A. Cron and
    M. West. 'Understanding GPU programming for statistical
    computation: Studies in massively parallel massive mixtures.'
    Journal of Computational and Graphical Statistics. 19 (2010):
    419-438

    Returns
    -------
    **Attributes**

    """

    def __init__(self, data, ncomp=256, gamma0=100, m0=None,
                 nu0=None, Phi0=None, e0=10, f0=1,
                 mu0=None, Sigma0=None, weights0=None, alpha0=1,
                 gpu=None, parallel=True, verbose=False):

        parallel = False ## Need to cythonize .... 
        ## for now, initialization is exactly the same .... 
        super(BEM_DPNormalMixture, self).__init__(
            data, ncomp, gamma0, m0, nu0, Phi0, e0, f0,
            mu0, Sigma0, weights0, alpha0, gpu, parallel,  verbose)
        self.alpha = self._alpha0
        self.weights = self._weights0.copy()
        self.stick_weights = self.weights.copy()
        self.mu = self._mu0.copy()
        self.Sigma = self._Sigma0.copy()
        self.e_labels = np.tile(self.weights.flatten(), (self.nobs, 1))
        self.densities = None

#     def __del__(self):
#         if self.parallel:
#             for thd in self.workers:
#                 thd.terminate()
        #if self.gpu:
        #    if hasattr(self, 'gpu_workers'):
        #        kill_GPUWorkers(self.gpu_workers)

    def optimize(self, maxiter=1000, perdiff=0.1):
        """
        Optimizes the posterior distribution given the data. The
        algorithm terminates when either the maximum number of
        iterations is reached or the percent difference in the
        posterior is less than perdiff.
        """

        #if self.gpu:
        #    self.gdata = to_gpu(np.asarray(self.data, dtype=np.float32))
        #    self.g_ones = to_gpu(np.ones((self.ncomp,1), dtype=np.float32))
        #    self.g_ones_long = to_gpu(np.ones((self.nobs, 1), dtype=np.float32))

#         if self.parallel:
#             from multiprocessing import RawArray
#             self.shared_dens_mem = RawArray('d', self.nobs*self.ncomp)
#             self.shared_dens = np.frombuffer(self.shared_dens_mem).reshape(self.nobs, self.ncomp)
#             for w in self.workers:
#                 w.set_dens(self.shared_dens_mem)
#                 w.start()

        # start threads
        if self.gpu:
            self.gpu_workers = init_GPUWorkers(self.data, self.dev_list)                


        self.expected_labels()
        ll_2 = self.log_posterior()
        ll_1 = 1
        it = 0
        if self.verbose:
            if self.gpu:
                print "starting GPU enabled BEM"
            else:
                print "starting BEM"
        while np.abs(ll_1 - ll_2) > 0.01*perdiff and it < maxiter:
            if isinstance(self.verbose, int) and self.verbose and not isinstance(self.verbose, bool):
                if it % self.verbose == 0:
                    print "%d:, %f" % (it, ll_2)
            it += 1

            self.maximize_mu()
            self.maximize_Sigma()
            self.maximize_weights()
            self.expected_alpha()
            self.expected_labels()
            ll_1 = ll_2
            ll_2 = self.log_posterior()
        if self.gpu:
            kill_GPUWorkers(self.gpu_workers)
#         if self.parallel:
#             for i in xrange(self.num_cores):
#                 self.work_queue[i].put(None)
                
    def log_posterior(self):
        # just the log likelihood right now because im lazy ... 
        return self.ll
    ll=0;
    _logmnflt = np.log(1e-37)
    def expected_labels(self):
        if self.gpu:
            self.ll, self.ct, self.xbar, self.densities = get_expected_labels_GPU(
                self.gpu_workers, self.weights, self.mu, self.Sigma)

        else:
            densities = mvn_weighted_logged(self.data, self.mu, self.Sigma, self.weights)
            densities = np.exp(densities)
            norm = densities.sum(1)
            self.ll = np.sum(np.log(norm))
            densities = (densities.T / norm).T
            self.ct = densities.sum(0)
            self.xbar = np.dot(densities.T, self.data)
            self.densities = densities

    def expected_alpha(self):
        
        sm = np.sum(np.log(1. - self.stick_weights[:-1]))
        self.alpha = (self.ncomp + self.e - 1.) / (self.f - sm)

    def maximize_mu(self):
        k, p = self.ncomp, self.ndim
        self.mu = (np.tile(self.mu_prior_mean, (k, 1)) + 
                   np.tile(self.gamma.reshape(k,1), (1,p))*self.xbar) / \
                   np.tile(( 1. + self.gamma * self.ct).reshape(k,1), (1,p))

    def maximize_Sigma(self):
        df = self.ct + self._nu0 + 2*self.ndim + 3

        for j in xrange(self.ncomp):
            if self.ct[j]>0.1:
                Xj_d = (self.data - self.xbar[j,:]/self.ct[j])
                SS = np.dot(Xj_d.T * self.densities[:,j].flatten(), Xj_d)
                SS += self._Phi0[j] + (self.ct[j]/(1+self.gamma[j]*self.ct[j]))*np.outer(
                    (1/self.ct[j])*self.xbar[j,:] - self.mu_prior_mean,
                    (1/self.ct[j])*self.xbar[j,:] - self.mu_prior_mean)
                self.Sigma[j] = SS / self.ct[j]

    def maximize_weights(self):
        sm = np.sum(self.ct)
        self.stick_weights = np.minimum(np.ones(len(self.stick_weights))-1e-10,
                                        self.ct / (self.alpha - 1 + 
                                                   self.ct[::-1].cumsum()[::-1]))
        self.stick_weights = np.maximum(np.ones(len(self.stick_weights))*1e-10, 
                                        self.stick_weights)
        self.stick_weights[-1]=1.

        V = self.stick_weights[:-1]
        pi = self.weights
        
        pi[0] = V[0]
        prod = (1 - V[0])
        for k in xrange(1, len(V)):
            pi[k] = prod * V[k]
            prod *= 1 - V[k]
        pi[-1] = prod


        
