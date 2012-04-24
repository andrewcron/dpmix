'''
Created on Mar 15, 2012

@author: Andrew Cron
'''
import numpy as np
import scipy.linalg as LA
import scipy.stats as stats

from utils import mvn_weighted_logged
from dpmix import DPNormalMixture

# check for gpustats compatability
try:
    import pycuda
    try:
        #import gpustats
        #import gpustats.sampler
        #import pycuda.gpuarray as gpuarray
        #from gpustats.util import GPUarray_reshape, GPUarray_order
        #from pycuda.gpuarray import to_gpu
        #from pycuda import cumath
        #from pycuda.elementwise import ElementwiseKernel
        #from scikits.cuda import linalg as cuLA; cuLA.init()
        #from cuda_functions import *
        #inplace_exp = ElementwiseKernel("float *z", "z[i]=expf(z[i])", "inplexp")
        #inplace_sqrt = ElementwiseKernel("float *z", "z[i]=sqrtf(z[i])", "inplsqrt")
        #gpu_copy = ElementwiseKernel("float *x, float *y", "x[i]=y[i]", "copyarraygpu")
        from multigpu import init_GPUWorkers, get_expected_labels_GPU, kill_GPUWorkers
        _has_gpu = True
    except (ImportError, pycuda._driver.RuntimeError):
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
                 nu0=None, Phi0=None, e0=1, f0=1,
                 mu0=None, Sigma0=None, weights0=None, alpha0=1,
                 gpu=None, verbose=False):

        ## for now, initialization is exactly the same .... 
        super(BEM_DPNormalMixture, self).__init__(
            data, ncomp, gamma0, m0, nu0, Phi0, e0, f0,
            mu0, Sigma0, weights0, alpha0, gpu, verbose)
        self.alpha = self._alpha0
        self.weights = self._weights0.copy()
        self.stick_weights = self.weights.copy()
        self.mu = self._mu0.copy()
        self.Sigma = self._Sigma0.copy()
        self.e_labels = np.tile(self.weights.flatten(), (self.nobs, 1))
        self.densities = None

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

        # start threads
        if self.gpu:
            for w in self.gpu_workers:
                w.start()
        if self.parallel:
            from multiprocessing import RawArray
            self.shared_dens_mem = RawArray('d', self.nobs*self.ncomp)
            self.shared_dens = np.frombuffer(self.shared_dens_mem).reshape(self.nobs, self.ncomp)
            for w in self.workers:
                w.set_dens(self.shared_dens_mem)
                w.start()
                


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
        if self.parallel:
            for i in xrange(self.num_cores):
                self.work_queue.put(None)
                
    def log_posterior(self):
        # just the log likelihood right now because im lazy ... 
        return self.ll
    ll=0;
    _logmnflt = np.log(1e-37)
    def expected_labels(self):
        if self.gpu:
            # densities = gpustats.mvnpdf_multi(self.gdata, self.mu, self.Sigma, 
            #                                   weights=self.weights.flatten(), 
            #                                   get=False, logged=True)
            # tdens = GPUarray_reshape(densities, (self.ncomp, self.nobs), "C")
            # #tdens = densities.reshape(self.ncomp, self.nobs, "C")
            # #import pdb; pdb.set_trace()
            # self.ll = cuLA.dot(self.g_ones, cumath.exp(tdens), "T").get()
            # nmzero = np.sum(self.ll==0)
            # self.ll = np.sum(np.log(self.ll[self.ll>0])) + nmzero*self._logmnflt

            # nrm, _ = gpu_apply_row_max(densities)
            # gpu_sweep_col_diff(densities, nrm)
            # inplace_exp(densities); GPUarray_order(densities, "F")
            # nrm = cuLA.dot(self.g_ones, tdens, "T")
            # gpu_sweep_col_div(densities, nrm)

            # self.ct = cuLA.dot(tdens, self.g_ones_long).get().flatten()
            # self.xbar = cuLA.dot(tdens, self.gdata).get()
            # self.densities = densities

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
        # if self.gpu:
        #     inplace_sqrt(self.densities); GPUarray_order(self.densities, "F")
        #     #fltdens = self.densities.ravel()
        #     fltdens = GPUarray_reshape(self.densities, self.densities.size)
        #     self.xbar = (self.xbar.T / self.ct).T
        #     for j in xrange(self.ncomp):
        #         if self.ct[j]>0.1:
        #             Xj_d = self.gdata._new_like_me(); gpu_copy(Xj_d, self.gdata);
        #             cdens = fltdens[(j*self.nobs):((j+1)*self.nobs)]
        #             gpu_sweep_row_diff(Xj_d, self.xbar[j,:].flatten())
        #             gpu_sweep_col_mult(Xj_d, cdens)
        #             SS = cuLA.dot(Xj_d, Xj_d, "T").get()
        #             SS += self._Phi0[j] + (self.ct[j]/(1+self.gamma[j]*self.ct[j]))*np.outer(
        #                 (1/self.ct[j])*self.xbar[j,:] - self.mu_prior_mean,
        #                 (1/self.ct[j])*self.xbar[j,:] - self.mu_prior_mean)
        #             self.Sigma[j] = SS / self.ct[j]
        # else:

        ## multithread? 
        if self.parallel:
            self.shared_dens[:] = self.densities
            for j in xrange(self.ncomp):
                self.work_queue.put(BEMSigmaUpdate(self.ct, self.xbar, self.Sigma[j], j))
            num_jobs = self.ncomp
            while num_jobs:
                result = self.result_queue.get()
                j = result.comp
                self.Sigma[j] = result.Sigma.copy()
                num_jobs -= 1
        else:
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


        
