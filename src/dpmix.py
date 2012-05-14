"""
Notes
-----
References
Ishwaran & James (2001) Gibbs Sampling Methods for Stick-Breaking Priors
"""
from __future__ import division

import numpy as np
import numpy.random as npr

from utils import mvn_weighted_logged, sample_discrete, _get_mask, stick_break_proc, _get_cost, select_gpu
from multicpu import CPUWorker, CompUpdate

from wishart import invwishartrand, invwishartrand_prec

import multiprocessing
import cython

try:
    from munkres import munkres
except ImportError:
    _has_munkres = False

# check for gpustats compatability
try:
    import pycuda
    import pycuda.driver
    try:
        from multigpu import init_GPUWorkers, get_labelsGPU, kill_GPUWorkers
        _has_gpu = True
    except (ImportError, pycuda.driver.RuntimeError):
        _has_gpu=False
except ImportError:
    _has_gpu = False


class DPNormalMixture(object):
    """
    MCMC sampling for Truncated Dirichlet Process Mixture of Normals

    Parameters
    ----------
    data : ndarray (nobs x ndim) or (BEM_)DPNormalMixture class
    ncomp : int
        Number of mixture components

    Notes
    -----
    y ~ \sum_{j=1}^J \pi_j {\cal N}(\mu_j, \Sigma_j)
    \pi ~ stickbreak(\alpha)
    \alpha ~ Ga(e, f)
    \mu_j ~ N(0, m\Sigma_j)
    \Sigma_j ~ IW(nu0 + 2, nu0 * \Phi_j)

    The defaults for the prior parameters are reasonable for
    standardized data. However, a careful analysis should always
    include careful choosing of priors.

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

    def __init__(self, data, ncomp=256, gamma0=10, m0=None,
                 nu0=None, Phi0=None, e0=10, f0=1,
                 mu0=None, Sigma0=None, weights0=None, alpha0=1,
                 gpu=None, parallel=True, verbose=False):
        if issubclass(type(data), DPNormalMixture):
            self.data = data.data
            self.nobs, self.ndim = self.data.shape
            self.ncomp = data.ncomp
            self.mu_prior_mean = data.mu_prior_mean
            nu0 = data._nu0
            Phi0 = data._Phi0
            if len(data.mu.shape)>2:
                mu0 = data.mu[-1].copy()
                Sigma0 = data.Sigma[-1].copy()
                weights0 = data.weights[-1].copy()
            else:
                mu0 = data.mu.copy()
                Sigma0 = data.Sigma.copy()
                weights0 = data.weights.copy()
            e0 = data.e
            f0 = data.f
            self.gamma = data.gamma
            self.gpu = data.gpu
            if self.gpu:
                self.dev_list = np.unique(data.dev_list)
            self.parallel = data.parallel
        else:
            if _has_gpu:
                self.dev_list = np.asarray((0), dtype=np.int); self.dev_list.shape=1
                if gpu is not None:
                    if type(gpu) is bool:
                        self.gpu = gpu
                    else:
                        self.gpu = True
                        self.dev_list = np.asarray(np.abs(gpu), dtype=np.int)
                        if self.dev_list.shape == ():
                            self.dev_list.shape = 1
                        self.dev_list = np.unique(self.dev_list)
                else:
                    self.gpu = True
            else:
                self.gpu = False

            self.data = np.asarray(data)
            self.nobs, self.ndim = self.data.shape
            self.ncomp = ncomp

            # TODO hyperparameters
            # prior mean for component means
            if m0 is not None:
                if len(m0)==self.ndim:
                    self.mu_prior_mean = m0.copy()
                elif len(m0)==1:
                    self.mu_prior_mean = m0*np.ones(self.ndim)
            else:
                self.mu_prior_mean = np.zeros(self.ndim)

            self.gamma = gamma0*np.ones(ncomp)
            self.parallel = parallel
                        
        #verbosity
        self.verbose = verbose
        
        self._set_initial_values(alpha0, nu0, Phi0, mu0, Sigma0,
                                 weights0, e0, f0)
        ## Check data for non-contiguous crap
        if not (self.data.flags["C_CONTIGUOUS"] or self.data.flags["F_CONTIGUOUS"]):
            self.data = self.data.copy()
        
        ## multiCPU stuf
        if self.parallel:
            self.num_cores = min(multiprocessing.cpu_count(), self.ncomp)
            compsperdev = self.ncomp / self.num_cores
            self.work_queue = [ multiprocessing.Queue() for i in xrange(self.num_cores) ]
            self.result_queue = [ multiprocessing.Queue() for i in xrange(self.num_cores) ]
            self.workers = [ CPUWorker(self.data, self.gamma, self.mu_prior_mean, 
                                       self._Phi0, self._nu0, self.work_queue[i], self.result_queue[i])
                             for i in xrange(self.num_cores) ]
            self.compsdevmap = {}; cumcomps = 0
            for i in xrange(self.num_cores):
                self.compsdevmap[i] = [int(cumcomps), int(min(cumcomps+compsperdev, self.ncomp))]
                cumcomps += compsperdev
            self.compsdevmap[self.num_cores-1][1] = self.ncomp

        
    def _set_initial_values(self, alpha0, nu0, Phi0, mu0, Sigma0, weights0,
                            e0, f0):
        if nu0 is None:
            nu0 = 1

        if Phi0 is None:
            Phi0 = np.empty((self.ncomp, self.ndim, self.ndim))
            Phi0[:] = np.eye(self.ndim) * nu0

        if Sigma0 is None:
            # draw from prior .. bad idea for vague prior ??? 
            Sigma0 = np.empty((self.ncomp, self.ndim, self.ndim))
            for j in xrange(self.ncomp):
                Sigma0[j] = invwishartrand(nu0 + 1 + self.ndim, Phi0[j])
            #Sigma0 = Phi0.copy()

        # starting values, are these sensible?
        if mu0 is None:
            mu0 = np.empty((self.ncomp, self.ndim))
            for j in xrange(self.ncomp):
                mu0[j] = npr.multivariate_normal(np.zeros((self.ndim)),
                                           self.gamma[j]*Sigma0[j])

        if weights0 is None:
            weights0 = (1/self.ncomp)*np.ones((self.ncomp, 1))
            #_, weights0 = stick_break_proc(1, 1, size=self.ncomp - 1)

        self._alpha0 = alpha0
        self.e = e0
        self.f = f0

        self._weights0 = weights0
        self._mu0 = mu0
        self._Sigma0 = Sigma0
        self._nu0 = nu0 # prior degrees of freedom
        self._Phi0 = Phi0 # prior location for Sigma_j's

    def sample(self, niter=1000, nburn=0, thin=1, ident=False):
        """
        samples niter + nburn iterations only storing the last niter
        draws thinned as indicated.

        if ident is True the munkres identification algorithm will be
        used matching to the INITIAL VALUES. These should be selected
        with great care. We recommend using the EM algorithm. Also
        .. burning doesn't make much sense in this case.
        """

        self._setup_storage(niter)

        # start threads
        if self.parallel:
            for w in self.workers:
                w.start()

        if self.gpu:
            self.gpu_workers = init_GPUWorkers(self.data, self.dev_list)

        alpha = self._alpha0
        weights = self._weights0
        mu = self._mu0
        Sigma = self._Sigma0

        if self.verbose:
            if self.gpu:
                print "starting GPU enabled MCMC"
            else:
                print "starting MCMC"

        for i in range(-nburn, niter):

            if i==0 and ident:
                labels, zref = self._update_labels(mu, Sigma, weights, True)
                c0 = np.zeros((self.ncomp, self.ncomp), dtype=np.double)
                for i in xrange(self.ncomp):
                    c0[i,:] = np.sum(zref==i)
                zhat = zref.copy()


            if isinstance(self.verbose, int) and self.verbose and \
                    not isinstance(self.verbose, bool):
                if i % self.verbose == 0:
                    print i

            labels, zhat = self._update_labels(mu, Sigma, weights, ident)
            mu, Sigma, counts = self._update_mu_Sigma(Sigma, labels)

            stick_weights, weights = self._update_stick_weights(counts, alpha)

            alpha = self._update_alpha(stick_weights)


            ## relabel if needed:
            if i>0 and ident:
                cost = c0.copy()
                _get_cost(zref, zhat, cost) #cython!!
                _, iii = np.where(munkres(cost))
                weights = weights[iii]
                mu = mu[iii]
                Sigma = Sigma[iii]
            if i>= 0:
                self.weights[i] = weights
                self.alpha[i] = alpha
                self.mu[i] = mu
                self.Sigma[i] = Sigma

        # clean up threads
        if self.parallel:
            for i in xrange(self.num_cores):
                self.work_queue[i].put(None)
        if self.gpu:
            kill_GPUWorkers(self.gpu_workers)

    # so pylint won't complain so much
    # alpha hyperparameters
    e = f = 1
    weights = None
    mu = None
    Sigma = None
    alpha = None
    stick_weights = None

    def _setup_storage(self, niter=1000, thin=1):
        nresults = niter // thin

        self.weights = np.zeros((nresults, self.ncomp))
        self.mu = np.zeros((nresults, self.ncomp, self.ndim))
        self.Sigma = np.zeros((nresults, self.ncomp, self.ndim, self.ndim))
        self.alpha = np.zeros(nresults)

    def _update_labels(self, mu, Sigma, weights, ident=False):
        if self.gpu:
            # GPU business happens?
	    #print self.data.shape, weights.shape, mu.shape, Sigma.shape
            return get_labelsGPU(self.gpu_workers, weights, mu, Sigma, relabel=ident) 
        else:
            densities = mvn_weighted_logged(self.data, mu, Sigma, weights)
            #self._densities = densities
            if ident:
                Z = densities.argmax(1)
            else:
                Z = None
            return sample_discrete(densities).squeeze(), Z

    def _update_stick_weights(self, counts, alpha):

        reverse_cumsum = counts[::-1].cumsum()[::-1]

        a = 1 + counts[:-1]
        b = alpha + reverse_cumsum[1:]
        stick_weights, mixture_weights = stick_break_proc(a, b)
        return stick_weights, mixture_weights

    def _update_alpha(self, V):
        a = self.ncomp + self.e - 1
        b = self.f - np.log(1 - V).sum()
        return npr.gamma(a, scale=1 / b)

    def _update_mu_Sigma(self, Sigma, labels, other_dat=None):
        is_hdp = isinstance(labels, list)

        mu_output = np.zeros((self.ncomp, self.ndim))
        Sigma_output = np.zeros((self.ncomp, self.ndim, self.ndim))
        if is_hdp:
            ct = np.zeros((len(labels), self.ncomp), dtype=np.int)
        else:
            ct = np.zeros(self.ncomp, dtype=np.int)

        if other_dat is None:
            data = self.data
        else:
            data = other_dat

        if self.parallel:
            num_jobs = len(self.compsdevmap)
            for tid in self.compsdevmap:
                rng = self.compsdevmap[tid]
                self.work_queue[tid].put(CompUpdate(np.arange(rng[0], rng[1]), labels, Sigma[rng[0]:rng[1]]))

            #while num_jobs:
            for tid in self.compsdevmap:
                result = self.result_queue[tid].get()
                newcomps = result.comps
                rng = (newcomps[0], newcomps[-1]+1)
                mu_output[rng[0]:rng[1]] = result.new_mu
                Sigma_output[rng[0]:rng[1]] = result.new_Sigma
                if is_hdp:
                    ct[:,rng[0]:rng[1]] = result.count
                else:
                    ct[rng[0]:rng[1]] = result.count
                num_jobs -= 1            

        else:

            for j in xrange(self.ncomp):
                if is_hdp:
                    nobs = data.shape[0]
                    mask = np.zeros(nobs, dtype=np.bool)
                    count = np.zeros(len(labels), dtype=np.int)
                    cumobs = 0; ii = 0
                    for labs in labels:
                        submask = labs == j
                        mask[cumobs:(cumobs+len(labs))] = submask
                        count[ii] = np.sum(submask); 
                        cumobs+=len(labs); ii+=1
                else:
                    mask = labels == j
                    count = np.sum(mask)


                Xj = data[mask]
                nj = len(Xj)
                
                sumxj = Xj.sum(0)

                gam = self.gamma[j]
                mu_hyper = self.mu_prior_mean

                post_mean = (mu_hyper / gam + sumxj) / (1 / gam + nj)
                post_cov = 1 / (1 / gam + nj) * Sigma[j]

                new_mu = npr.multivariate_normal(post_mean, post_cov)
                
                Xj_demeaned = Xj - new_mu

                mu_SS = np.outer(new_mu - mu_hyper, new_mu - mu_hyper) / gam
                data_SS = np.dot(Xj_demeaned.T, Xj_demeaned)
                post_Phi = data_SS + mu_SS + self._Phi0[j]

                # symmetrize
                post_Phi = (post_Phi + post_Phi.T) / 2
                
                # P(Sigma) ~ IW(nu + 2, nu * Phi)
                # P(Sigma | theta, Y) ~
                post_nu = nj + self.ndim + self._nu0 + 2

                new_Sigma = invwishartrand_prec(post_nu, post_Phi)

                mu_output[j] = new_mu
                Sigma_output[j] = new_Sigma
                if is_hdp:
                    ct[:, j] = count
                else:
                    ct[j] = count



        return mu_output, Sigma_output, ct




