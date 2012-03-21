from __future__ import division

import numpy as np
import numpy.random as npr
from scipy import stats

import pymc as pm

from utils import mvn_weighted_logged, sample_discrete, _get_mask, stick_break_proc, _get_cost
from utils import break_sticks
from dpmix import DPNormalMixture

import cython

import pdb

try:
    from munkres import munkres
except ImportError:
    _has_munkres = False

# check for gpustats compatability
try:
    import pycuda
    try:
        import gpustats
        import gpustats.sampler
        import pycuda.gpuarray as gpuarray
        from pycuda.gpuarray import to_gpu
        from pycuda import cumath
        from pycuda.elementwise import ElementwiseKernel
        from scikits.cuda import linalg as cuLA; cuLA.init()
        from cuda_functions import *
        inplace_exp = ElementwiseKernel("float *z", "z[i]=expf(z[i])", "inplexp")
        inplace_sqrt = ElementwiseKernel("float *z", "z[i]=sqrtf(z[i])", "inplsqrt")
        gpu_copy = ElementwiseKernel("float *x, float *y", "x[i]=y[i]", "copyarraygpu")
        _has_gpu = True
    except (ImportError, pycuda._driver.RuntimeError):
        _has_gpu=False
except ImportError:
    _has_gpu = False

# func to get lpost for beta
@cython.compile
def beta_post(stick_beta, beta, stick_weights, alpha0, alpha):
    J = stick_weights.shape[0]
    k = stick_weights.shape[1]
    lpost = 0
    for j in xrange(J):
        a, b = alpha0*beta[:-1], alpha0*(1-beta[:-1].cumsum())
        lpost += np.sum(stats.beta.logpdf(stick_weights, a, b))
    lpost += np.sum(stats.beta.logpdf(stick_beta, 1, alpha))
    return lpost

class HDPNormalMixture(DPNormalMixture):
    """
    MCMC sampling for Doubly Truncated HDP Mixture of Normals for multiple datasets

    Parameters
    -----------
    data :  list of ndarrays (nobs x ndim) -- ndim must be equal .. not nobs
    ncomp : nit
        Number of mixture components

    Returns
    -------
    **Attributes**
    """

    def __init__(self, data, ncomp=256, gamma0=10, m0=None,
                 nu0=None, Phi0=None, e0=1, f0=1, g0=1, h0=1, 
                 mu0=None, Sigma0=None, weights0=None, alpha0=1,
                 gpu=None):

        # check for functioning gpu
        if _has_gpu:
            if gpu is not None:
                self.gpu = gpu
            else:
                self.gpu = _has_gpu
        else:
            self.gpu = False
            
        # get the data .. should add checks here later
        self.data = [np.asarray(d) for d in data]
        self.ngroups = len(self.data)
        self.ndim = self.data[0].shape[1]
        self.nobs = tuple([d.shape[0] for d in self.data])
        self.ncomp = ncomp

        if m0 is not None:
            if len(m0)==self.ndim:
                self.mu_prior_mean = m0.copy()
            elif len(m0)==1:
                self.mu_prior_mean = m0*np.ones(self.ndim)
        else:
            self.mu_prior_mean = np.zeros(self.ndim)

        self.gamma = gamma0*np.ones(ncomp)
                        
        # set gpu working vars
        if self.gpu:
            self.gdata = [to_gpu(np.asarray(dat, dtype=np.float32)) for dat in self.data]

                
        self._set_initial_values(alpha0, nu0, Phi0, mu0, Sigma0,
                                 weights0, e0, f0)
        # initialize hdp specific vars
        self._weights0 = np.zeros((self.ngroups, self.ncomp), dtype=np.float)
        self._weights0.fill(1/self.ncomp)
        self._stick_beta0 = stats.beta.rvs(1,self._alpha0, size=self.ncomp-1)
        self._beta0 = break_sticks(self._stick_beta0)
        self.e0, self.f0 = f0, g0
        self.prop_scale = 0.05 * np.ones(self.ncomp)
        self.AR = np.zeros(self.ncomp)
        

    def sample(self, niter=1000, nburn=0, thin=1, tune_interval=100, ident=False):
        self._setup_storage(niter, thin)
        self._tune_interval = tune_interval

        alpha = self._alpha0
        alpha0 = 1.0
        weights = self._weights0
        beta = self._beta0
        stick_beta = self._stick_beta0
        mu = self._mu0
        Sigma = self._Sigma0

        for i in range(-nburn, niter):
            labels = self._update_labels(mu, Sigma, weights)
            component_mask = [ _get_mask(l, self.ncomp) for l in labels ]
            counts = [ mask.sum(1) for mask in component_mask ]
            stick_weights, weights = self._update_stick_weights(counts, beta, alpha0)
            try:
                stick_beta, beta = self._update_beta(stick_beta, beta, stick_weights, alpha0, alpha)
            except:
                pdb.set_trace()

            alpha = self._update_alpha(stick_beta)
            alpha0 = self._update_alpha0(stick_weights, beta, alpha0)

            mu, Sigma = self._update_mu_Sigma(mu, component_mask)

            if i>=0:
                self.beta[i] = beta
                self.weights[i] = weights
                self.alpha[i] = alpha
                self.alpha0[i] = alpha0
                self.mu[i] = mu
                self.Sigma[i] = Sigma
            elif (nburn+i+1)%self._tune_interval == 0:
                self._tune()

            

    def _setup_storage(self, niter=1000, thin=1):
        nresults = niter // thin
        self.weights = np.zeros((nresults, self.ngroups, self.ncomp))
        self.beta = np.zeros((nresults, self.ncomp))
        self.mu = np.zeros((nresults, self.ncomp, self.ndim))
        self.Sigma = np.zeros((nresults, self.ncomp, self.ndim, self.ndim))
        self.alpha = np.zeros(nresults)
        self.alpha0 = np.zeros(nresults)

    def _update_labels(self, mu, Sigma, weights):
        # gets the latent classifications .. 
        labels = [np.zeros(self.nobs[j]) for j in range(self.ngroups)]
        if self.gpu:
            for j in xrange(self.ngroups):
                densities = gpustats.mvnpdf_multi(self.gdata[j], mu, Sigma,
                                                  weights=weights[j].flatten(),
                                                  get=False, logged=True, order='C')
                labels[j] = gpustats.sampler.sample_discrete(densities, logged=True)
        else:
            for j in xrange(self.ngroups):
                densities = mvn_weighted_logged(self.data[j], mu, Sigma, weights[j])
                labels[j] = sample_discrete(densities).squeeze()
        return labels

    def _update_stick_weights(self, counts, beta, alpha0):
        new_weights = np.zeros((self.ngroups, self.ncomp))
        new_stick_weights = np.zeros((self.ngroups, self.ncomp-1))
        for j in xrange(self.ngroups):
            reverse_cumsum = counts[j][::-1].cumsum()[::-1]
            
            a = alpha0*beta[:-1] + counts[j][:-1]
            b = alpha0*(1-beta[:-1].cumsum()) + reverse_cumsum[1:]
            sticksj, weightsj = stick_break_proc(a, b)
            new_weights[j] = weightsj
            new_stick_weights[j] = sticksj
        return new_stick_weights ,new_weights

    def _update_beta(self, stick_beta, beta, stick_weights, alpha0, alpha):                
        old_stick_beta = stick_beta.copy()
        old_beta = beta.copy()
        for k in xrange(self.ncomp-1):
            # get initial logpost
            lpost = beta_post(stick_beta, beta, stick_weights, float(alpha0), float(alpha))
            
            # sample new beta from reflected normal
            prop = stats.norm.rvs(stick_beta[k], self.prop_scale[k])
            while prop > 1 or prop < 0:
                if prop > 1:
                    prop = 2 - prop
                else:
                    prop = - prop
            stick_beta[k] = prop
            beta = break_sticks(stick_beta)

            # get new posterior
            lpost_new = beta_post(stick_beta, beta, stick_weights, float(alpha0), float(alpha))

            # accept or reject
            if stats.expon.rvs() > lpost - lpost_new:
                #accept
                self.AR[k] += 1
            else:
                stick_beta[k] = old_stick_beta[k]
                beta = break_sticks(stick_beta)
        return stick_beta, beta
        
    def _update_alpha0(self, stick_weights, beta, alpha0):
        # just reuse with dummy vars for beta things
        lpost = beta_post(0.5*np.ones_like(beta), beta, stick_weights, float(alpha0), float(1))
        lpost += stats.gamma.logpdf(alpha0, self.e0, loc=0, scale=1.0/self.f0)
        alpha0_old = alpha0
        alpha0 = np.abs(stats.norm.rvs(alpha0, self.prop_scale[-1]))
        lpost_new = beta_post(0.5*np.ones_like(beta), beta, stick_weights, float(alpha0), float(1))
        lpost_new += stats.gamma.logpdf(alpha0, self.e0, loc=0, scale=1.0/self.f0)
        #accept or reject
        if stats.expon.rvs() > lpost - lpost_new:
            self.AR[-1] += 1
        else:
            alpha0 = alpha0_old
        return alpha0
        

    def _tune(self):
        """
        Rate    Variance adaptation
        ----    -------------------
        <0.001        x 0.1
        <0.05         x 0.5
        <0.2          x 0.9
        >0.5          x 1.1
        >0.75         x 2
        >0.95         x 10
        """
        for j in xrange(len(self.AR)):
            ratio = self.AR[j] / self._tune_interval
            if ratio < 0.001:
                self.prop_scale[j] *= np.sqrt(0.1)
            elif ratio < 0.05:
                self.prop_scale[j] *= np.sqrt(0.5)
            elif ratio < 0.2:
                self.prop_scale[j] *= np.sqrt(0.9)
            elif ratio > 0.95:
                self.prop_scale[j] *= np.sqrt(10)
            elif ratio > 0.75:
                self.prop_scale[j] *= np.sqrt(2)
            elif ratio > 0.5:
                self.prop_scale[j] *= np.sqrt(1.1)
            self.AR[j] = 0


    def _update_mu_Sigma(self, mu, masks):
        mu_output = np.zeros((self.ncomp, self.ndim))
        Sigma_output = np.zeros((self.ncomp, self.ndim, self.ndim))

        for j in xrange(self.ncomp):
            # get summary stats across multiple datasets
            sumxj = np.zeros(self.ndim)
            data_SS = np.zeros((self.ndim, self.ndim))
            nj = 0
            for d in xrange(self.ngroups):
                mask = masks[d][j]
                Xj = self.data[d][mask]
                nj += len(Xj)
                sumxj += Xj.sum(0)
                Xj_demeaned = Xj - mu[j]
                data_SS += np.dot(Xj_demeaned.T, Xj_demeaned)

            #update Sigma then mu
            mu_hyper = self.mu_prior_mean
            gam = self.gamma[j]

            mu_SS = np.outer(mu[j] - mu_hyper, mu[j] - mu_hyper) / gam
            post_Phi = data_SS + mu_SS + self._Phi0[j]
            # symmetrize just in case
            post_Phi = (post_Phi + post_Phi.T)/2
            post_nu = nj + self.ndim + self._nu0 + 2
            new_Sigma = pm.rinverse_wishart_prec(post_nu, post_Phi)
            #mu
            post_mean = (mu_hyper / gam + sumxj) / (1 / gam + nj)
            post_cov = 1 / (1 / gam + nj) * new_Sigma
            new_mu = pm.rmv_normal_cov(post_mean, post_cov)

            mu_output[j] = new_mu
            Sigma_output[j] = new_Sigma

        return mu_output, Sigma_output


            
            
            




            
