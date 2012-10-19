import numpy as np
cimport numpy as np
cimport cython

from libc.stdlib cimport calloc, malloc, free
from libc.math cimport sqrt


include "cyarma.pyx"
include "random.pyx"

@cython.boundscheck(False)
cdef mat wishartrand(double nu, mat phi, rng_sampler[double] rng) nogil:
    cdef int dim = phi.n_rows
    cdef mat phi_chol = chol(phi)

    cdef double *foo = <double*>calloc(dim*dim, sizeof(double))
    #cdef double[:,:] foo = <double[:dim,:dim]> cfoo

    cdef int i,j
    for i in range(dim):
        for j in range(i+1):
            if i == j:
                foo[i*dim+j] = sqrt(rng.chisq(nu-<double>i))
            else:
                foo[i*dim+j] = rng.normal(0, 1)
                

    cdef mat *mfoo = new mat(foo, dim, dim, False, True)
    cdef mat tmp =  deref(mfoo) * phi_chol

    free(foo)
    return tmp.t() * tmp

cdef mat invwishartrand(double nu, mat phi, rng_sampler[double] rng) nogil:
    return inv(wishartrand(nu, phi, rng))

cdef mat invwishartrand_prec(double nu, mat phi, rng_sampler[double] rng) nogil:
    return inv(wishartrand(nu, inv(phi), rng))

@cython.boundscheck(False)
cdef vec mvnormrand_eigs(vec mu, vec evals, mat evecs, rng_sampler[double] rng) nogil:
    # initialize
    cdef int i
    cdef int dim = mu.n_elem
    cdef vec * samp_pt = new vec(dim)
    cdef vec samp = deref(samp_pt)
    # sample normals
    for i in range(dim):
        samp[i] = rng.normal(0,1)
        evals[i] = sqrt(evals[i])

    samp = samp % evals
    samp = evecs * samp
    samp = samp + mu
    return samp

cdef vec mvnormrand(vec mu, mat Sigma, rng_sampler[double] rng) nogil:
    cdef int dim = mu.n_elem
    cdef vec * evals_pt = new vec(dim)
    cdef vec evals = deref(evals_pt)
    cdef mat * evecs_pt = new mat(dim, dim)
    cdef mat evecs = deref(evecs_pt)
    eig_sym(evals, evecs, Sigma)
    return mvnormrand_eigs(mu, evals, evecs, rng)

@cython.boundscheck(False)
@cython.cdivision(True)
cdef vec mvnormrand_prec(vec mu, mat Tau, rng_sampler[double] rng) nogil:
    cdef int dim= mu.n_elem
    cdef int i 
    cdef vec * evals_pt = new vec(dim)
    cdef vec evals = deref(evals_pt)
    cdef mat * evecs_pt = new mat(dim, dim)
    cdef mat evecs = deref(evecs_pt)
    eig_sym(evals, evecs, Tau)
    for i in range(dim):
        evals[i] = 1/evals[i]
    return mvnormrand_eigs(mu, evals, evecs, rng)

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _sample_component(vec & mu, mat & Sigma,
                            mat & data, vec & labels, int cur_lab,
                            vec & count,
                            double gamma, vec & pr_mu,
                            double & pr_nu, mat & pr_phi,
                            rng_sampler[double] & rng) nogil:
    ## this samples the normal paremters for one component
    
    # initialize working memory
    cdef int i,j,k
    cdef double diff
    cdef int n = data.n_cols
    cdef int p = data.n_rows
    cdef int nj = 0
    cdef vec * sumxj_p = new vec(p)
    cdef vec sumxj = deref(sumxj_p)
    cdef mat * SS_p = new mat(p,p)
    cdef mat SS = deref(SS_p)
    for i in range(p):
        sumxj[i] = 0
        for j in range(p):
            SS[j+i*p] = 0

    # get sufficient statistics for data subset
    for i in range(n):
        if labels[i] == cur_lab:
            nj += 1
            for j in range(p):
                sumxj[j] += data[j + i*p]
                for k in range(p):
                    SS[k+j*p] += (data[j+i*p]-mu[j])*(data[k+i*p]-mu[k])
    ## sample Sigma!
    for j in range(p): # gets prior contribution
        diff = mu[j] - pr_mu[j]
        for k in range(p):
            SS[k+j*p] += (1/gamma) * diff * (mu[k] - pr_mu[k])
            SS[k+j*p] += pr_phi[k+j*p]
    cdef mat new_Sigma = invwishartrand_prec(nj+p+pr_nu+2, SS, rng)
    ## copy sampled Sigma over current sigma
    for j in range(p):
        for k in range(p):
            Sigma[k+j*p] = new_Sigma[k+j*p]
    ## sample mu!!
    cdef vec post_mean = ((pr_mu / gamma) + sumxj) / (1 / gamma + <double>nj)
    cdef mat post_cov = Sigma  / (1 / gamma + <double>nj) 
    cdef vec new_mu = mvnormrand(post_mean, post_cov, rng)
    ## copy sample mu over currect mu
    for j in range(p):
        mu[j] = new_mu[j]
    
    count[cur_lab] = <double>nj

## CREATE SAFE VIEWS!!!
@cython.boundscheck(False)
@cython.cdivision(True)
cdef vec __sample_mu_Sigma(mat & mu, cube & Sigma, vec & labels, mat & data,
                           double gamma, vec & pr_mu, double pr_nu, mat & pr_phi):
    cdef int k
    cdef rng_sampler[double] * R = new rng_sampler[double]()
    
    cdef int p = mu.n_rows
    cdef int ncomp = mu.n_cols

    cdef vec * count_p = new vec(ncomp)
    cdef vec count = deref(count_p)

    cdef vec * cmu
    cdef mat * cSigma

    for k in range(ncomp):
        cSigma = cube_slice_view(Sigma, k)
        cmu = mat_col_view(mu,k)
        _sample_component(deref(cmu), deref(cSigma),
                          data, labels, k, count,
                          gamma, pr_mu,
                          pr_nu, pr_phi, deref(R))
        
    return count

def sample_mu_Sigma(np.ndarray[np.double_t, ndim=2] mu_in,
                    np.ndarray[np.double_t, ndim=3] Sigma_in,
                    np.ndarray[np.int_t, ndim=1] labels_in,
                    np.ndarray[np.double_t, ndim=2] data_in,
                    double gamma, np.ndarray[np.double_t, ndim=1] pr_mu_in,
                    double pr_nu, np.ndarray[np.double_t, ndim=2] pr_phi_in):

    ## this code just handles type casting
    ## will eventually handle hdp and parallel issues

    ## mu_in and Sigma_in are updated in place
    
    ## convert to armadillo
    cdef mat * mu = numpy_to_mat(mu_in.T)
    cdef cube * Sigma = numpy_to_cube(Sigma_in)
    cdef vec * labels = numpy_to_vec(np.array(labels_in, dtype=np.double))
    cdef mat * data = numpy_to_mat(data_in.T)
    cdef vec * pr_mu = numpy_to_vec(pr_mu_in)
    cdef mat * pr_phi = numpy_to_mat(pr_phi_in.T)
    

    ## call
    cdef vec ct = __sample_mu_Sigma(deref(mu), deref(Sigma), deref(labels), deref(data),
                           gamma, deref(pr_mu), pr_nu, deref(pr_phi))

    return np.array(vec_to_numpy(ct,None),dtype=np.int)


# ############ some python functions for testing ############

def pyiwishartrand(double nu, np.ndarray[np.double_t, ndim=2] Phi):
    #cdef rng r
    cdef rng_sampler[double] * R = new rng_sampler[double]()
    cdef mat result = invwishartrand(nu, deref(numpy_to_mat(Phi)), deref(R))
    return mat_to_numpy(result, None)

def pyiwishartrand_prec(double nu, np.ndarray[np.double_t, ndim=2] Phi):
    #cdef rng r
    cdef rng_sampler[double] * R = new rng_sampler[double]()
    cdef mat result = invwishartrand_prec(nu, deref(numpy_to_mat(Phi)), deref(R))
    return mat_to_numpy(result, None)

def pymvnorm(np.ndarray[np.double_t, ndim=1] mu,
              np.ndarray[np.double_t, ndim=2] Sigma):

    cdef rng_sampler[double] * R = new rng_sampler[double]()
    cdef vec result = mvnormrand(numpy_to_vec(mu), deref(numpy_to_mat(Sigma)), deref(R))
    return vec_to_numpy(result, None)

def pymvnorm_prec(np.ndarray[np.double_t, ndim=1] mu,
              np.ndarray[np.double_t, ndim=2] Sigma):

    cdef rng_sampler[double] * R = new rng_sampler[double]()
    cdef vec result = mvnormrand_prec(numpy_to_vec(mu), deref(numpy_to_mat(Sigma)), deref(R))
    return vec_to_numpy(result, None)




