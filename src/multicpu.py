import multiprocessing
import pymc as pm
import numpy as np

######### Multi CPU for comp updates ##############
class CPUWorker(multiprocessing.Process):
    """
    CPU multiprocess class that only loads data once
    """ 
    def __init__(self, data, gamma, mu_prior_mean, Phi0, nu0, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.gamma = gamma.copy()
        self.mu_prior_mean = mu_prior_mean.copy()
        self.Phi0 = Phi0.copy()
        self.nu0 = nu0
        self.data = data.copy()
        self.nobs, self.ndim = self.data.shape
        self.ncomp = len(gamma)

    def set_dens(self, shared_dens):
        self.dens = np.frombuffer(shared_dens).reshape(self.nobs, self.ncomp)

    def run(self):
        while True:
            new_work = self.task_queue.get()
            if new_work is None:
                break # poison pill
            if type(new_work) == CompUpdate:
                new_work(self.data, self.gamma, self.mu_prior_mean, self.Phi0, self.nu0)
            else:
                new_work(self.data, self.gamma, self.mu_prior_mean, 
                         self.Phi0, self.nu0, self.dens)                
            self.result_queue.put(new_work)

class BEMSigmaUpdate(object):
    def __init__(self, ct, xbar, Sigma, comp):
        self.comp = comp
        self.ct = ct
        self.xbar = xbar
        self.Sigma = Sigma.copy()

    def __call__(self, data, gamma, mu_prior_mean, Phi0, nu0, dens):
        j = self.comp
        if self.ct[j]>0.1:
            Xj_d = (data - self.xbar[j,:]/self.ct[j])
            SS = np.dot(Xj_d.T * dens[:,j].flatten(), Xj_d)
            SS += Phi0[j] + (self.ct[j]/(1+gamma[j]*self.ct[j]))*np.outer(
                (1/self.ct[j])*self.xbar[j,:] - mu_prior_mean,
                (1/self.ct[j])*self.xbar[j,:] - mu_prior_mean)
            self.Sigma = SS / self.ct[j]


    

class CompUpdate(object):
    def __init__(self, comp, mask, Sigma):
        self.comp = comp
        self.mask = mask
        self.Sigma = Sigma

    def __call__(self, data, gamma, mu_prior_mean, Phi0, nu0):
        j = self.comp
        Xj = data[self.mask]
        nj, ndim = Xj.shape

        sumxj = Xj.sum(0)

        gam = gamma[j]
        mu_hyper = mu_prior_mean

        post_mean = (mu_hyper / gam + sumxj) / (1 / gam + nj)
        post_cov = 1 / (1 / gam + nj) * self.Sigma

        new_mu = pm.rmv_normal_cov(post_mean, post_cov)

        Xj_demeaned = Xj - new_mu

        mu_SS = np.outer(new_mu - mu_hyper, new_mu - mu_hyper) / gam
        data_SS = np.dot(Xj_demeaned.T, Xj_demeaned)
        post_Phi = data_SS + mu_SS + Phi0[j]

        # symmetrize
        post_Phi = (post_Phi + post_Phi.T) / 2

        # P(Sigma) ~ IW(nu + 2, nu * Phi)
        # P(Sigma | theta, Y) ~
        post_nu = nj + ndim + nu0 + 2

        # pymc rinverse_wishart takes
        new_Sigma = pm.rinverse_wishart_prec(post_nu, post_Phi)

        # store new results in class
        self.new_Sigma = new_Sigma
        self.new_mu = new_mu

