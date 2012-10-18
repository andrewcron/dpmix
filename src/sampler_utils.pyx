import numpy as np
cimport numpy as np
cimport cython

from libc.stdlib cimport calloc, malloc, free
from libc.math cimport sqrt


include "cyarma.pyx"
include "random.pyx"

@cython.boundscheck(False)
cdef mat wishartrand(double nu, mat phi, rng_sampler[double] rng):
    cdef int dim = phi.n_rows
    cdef mat phi_chol = chol(phi)

    cdef double *cfoo = <double*>calloc(dim*dim, sizeof(double))
    cdef double[:,:] foo = <double[:dim,:dim]> cfoo

    cdef int i,j
    for i in range(dim):
        for j in range(i+1):
            if i == j:
                foo[i,j] = sqrt(rng.chisq(nu-<double>i))
            else:
                foo[i,j] = rng.normal(0, 1)
                

    cdef mat *mfoo = new mat(cfoo, dim, dim, False, True)
    cdef mat tmp =  deref(mfoo) * phi_chol

    free(cfoo)
    return tmp.t() * tmp

cdef mat invwishartrand(double nu, mat phi, rng_sampler[double] rng):
    return inv(wishartrand(nu, phi, rng))

cdef mat invwishartrand_prec(double nu, mat phi, rng_sampler[double] rng):
    return inv(wishartrand(nu, inv(phi), rng))

@cython.boundscheck(False)
cdef vec mvnormrand_eigs(vec mu, vec evals, mat evecs, rng_sampler[double] rng):
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

cdef vec mvnormrand(vec mu, mat Sigma, rng_sampler[double] rng):
    cdef int dim = mu.n_elem
    cdef vec * evals_pt = new vec(dim)
    cdef vec evals = deref(evals_pt)
    cdef mat * evecs_pt = new mat(dim, dim)
    cdef mat evecs = deref(evecs_pt)
    eig_sym(evals, evecs, Sigma)
    return mvnormrand_eigs(mu, evals, evecs, rng)

@cython.boundscheck(False)
@cython.cdivision(True)
cdef vec mvnormrand_prec(vec mu, mat Tau, rng_sampler[double] rng):
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


def pyiwishartrand(double nu, np.ndarray[np.double_t, ndim=2] Phi):
    #cdef rng r
    cdef rng_sampler[double] * R = new rng_sampler[double]()
    cdef mat result = invwishartrand(nu, numpy_to_mat(Phi), deref(R))
    return mat_to_numpy(result, None)

def pyiwishartrand_prec(double nu, np.ndarray[np.double_t, ndim=2] Phi):
    #cdef rng r
    cdef rng_sampler[double] * R = new rng_sampler[double]()
    cdef mat result = invwishartrand_prec(nu, numpy_to_mat(Phi), deref(R))
    return mat_to_numpy(result, None)

def pymvnorm(np.ndarray[np.double_t, ndim=1] mu,
              np.ndarray[np.double_t, ndim=2] Sigma):

    cdef rng_sampler[double] * R = new rng_sampler[double]()
    cdef vec result = mvnormrand(numpy_to_vec(mu), numpy_to_mat(Sigma), deref(R))
    return vec_to_numpy(result, None)

def pymvnorm_prec(np.ndarray[np.double_t, ndim=1] mu,
              np.ndarray[np.double_t, ndim=2] Sigma):

    cdef rng_sampler[double] * R = new rng_sampler[double]()
    cdef vec result = mvnormrand_prec(numpy_to_vec(mu), numpy_to_mat(Sigma), deref(R))
    return vec_to_numpy(result, None)




