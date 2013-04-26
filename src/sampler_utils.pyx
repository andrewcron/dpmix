import numpy as np
cimport numpy as np
cimport cython
cimport openmp
from cython.parallel import prange, parallel, threadid
import multiprocessing

from libc.stdlib cimport calloc, malloc, free
from libc.math cimport sqrt, lgamma, log, fabs


include "cyarma.pyx"
include "random.pyx"

# hack to get barrier (sync) for open mp
cdef extern from *:
    int OMP_BARRIER "#pragma omp barrier\n"

cdef extern from "<vector>" namespace "std" nogil:
    cdef cppclass vector[T]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        vector() nogil
        void push_back(T&) nogil
        T& operator[](int) nogil
        T& at(int) nogil
        iterator begin() nogil
        iterator end() nogil
        void erase(iterator) nogil

@cython.boundscheck(False)
cdef mat wishartrand(double nu, mat phi, rng_sampler[double] & rng) nogil:
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
    del mfoo
    return tmp.t() * tmp

cdef mat invwishartrand(double nu, mat & phi, rng_sampler[double] & rng)  nogil:
    return inv(wishartrand(nu, phi, rng))

cdef mat invwishartrand_prec(double nu, mat & phi, rng_sampler[double] & rng)  nogil:
    return inv(wishartrand(nu, inv(phi), rng))

cdef mat wishartrand_prec(double nu, mat & phi, rng_sampler[double] & rng)  nogil:
    return wishartrand(nu, inv(phi), rng)


@cython.boundscheck(False)
cdef vec mvnormrand_eigs(vec & mu, vec & evals, mat & evecs, rng_sampler[double] & rng) nogil:
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
    del samp_pt
    return samp

cdef vec mvnormrand(vec & mu, mat & Sigma, rng_sampler[double] & rng)  nogil:
    cdef int dim = mu.n_elem
    cdef vec * evals_pt = new vec(dim)
    cdef vec evals = deref(evals_pt)
    cdef mat * evecs_pt = new mat(dim, dim)
    cdef mat evecs = deref(evecs_pt)
    eig_sym(evals, evecs, Sigma)
    cdef vec result = mvnormrand_eigs(mu, evals, evecs, rng)
    del evals_pt
    del evecs_pt
    return result

@cython.boundscheck(False)
@cython.cdivision(True)
cdef vec mvnormrand_prec(vec & mu, mat & Tau, rng_sampler[double] & rng)  nogil:
    cdef int dim= mu.n_elem
    cdef int i 
    cdef vec * evals_pt = new vec(dim)
    cdef vec evals = deref(evals_pt)
    cdef mat * evecs_pt = new mat(dim, dim)
    cdef mat evecs = deref(evecs_pt)
    eig_sym(evals, evecs, Tau)
    for i in range(dim):
        evals[i] = 1/evals[i]
    cdef vec result = mvnormrand_eigs(mu, evals, evecs, rng)
    del evals_pt
    del evecs_pt
    return result

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _sample_component(vec & mu, mat & Sigma,
                            mat & data, 
                            double count, vector[int] & mask,
                            double gamma, vec & pr_mu,
                            double pr_nu, mat & pr_phi,
                            rng_sampler[double] & rng) nogil:
    ## this samples the normal paremters for one component

    # initialize working memory
    cdef int i,j,k,ii,nj
    cdef double diff
    cdef int n = data.n_cols
    cdef int p = data.n_rows
    cdef vec * sumxj_p = new vec(p)
    cdef vec sumxj = deref(sumxj_p)
    cdef mat * SS_p = new mat(p,p)
    cdef mat SS = deref(SS_p)
    for i in range(p):
        sumxj[i] = 0
        for j in range(p):
            SS[j+i*p] = 0

    # get sufficient statistics for data subset (if any data)
    nj = <int>count
    if nj > 0:
        for i in range(nj):
            ii = mask.at(i)
            for j in range(p):
                sumxj[j] += data[j + ii*p]
                diff = data[j+ii*p]-mu[j]
                for k in range(p):
                    SS[k+j*p] += diff*(data[k+ii*p]-mu[k])
    ## sample Sigma!
    for j in range(p): # gets prior contribution
        diff = mu[j] - pr_mu[j]
        for k in range(p):
            SS[k+j*p] += (1/gamma) * diff * (mu[k] - pr_mu[k])
            SS[k+j*p] += pr_phi[k+j*p]
    ## take advantage of free inverse
    #cdef mat new_Sigma = invwishartrand_prec(nj+p+pr_nu+2, SS, rng)
    cdef mat new_iSigma = wishartrand_prec(nj+p+pr_nu+2, SS, rng)
    cdef vec * evals_pt = new vec(p)
    cdef vec evals = deref(evals_pt)
    cdef mat * evecs_pt = new mat(p, p)
    cdef mat evecs = deref(evecs_pt)
    eig_sym(evals, evecs, new_iSigma)
    for j in range(p):
        evals[j] = 1.0/evals[j]
    
    # get Sigma smart like ... or only keep inverse!!!! more work ...
    cdef mat * new_Sigma_pt = new mat(p,p)
    cdef mat new_Sigma = deref(new_Sigma_pt)
    for j in range(p):
        for i in range(j,p):
            new_Sigma[i+j*p] = 0
            for k in range(p):
                new_Sigma[i+j*p] += evals[k]*evecs[i+k*p]*evecs[j+k*p]
    for j in range(1,p):
        for i in range(j):
            new_Sigma[i+j*p] = new_Sigma[j+i*p]
    
    
    ## copy sampled Sigma over current sigma
    for j in range(p):
        for k in range(p):
            Sigma[k+j*p] = new_Sigma[k+j*p]
    ## sample mu!!
    cdef vec post_mean = ((pr_mu / gamma) + sumxj) / (1 / gamma + <double>nj)
    #cdef mat post_cov = Sigma / (1 / gamma + <double>nj)
    evals = evals / (1/gamma + <double>nj)
    #cdef vec new_mu = mvnormrand(post_mean, post_cov, rng)
    cdef vec new_mu = mvnormrand_eigs(post_mean, evals, evecs, rng)
    ## copy sample mu over currect mu
    for j in range(p):
        mu[j] = new_mu[j]

    del sumxj_p
    del SS_p
    del new_Sigma_pt
    del evals_pt
    del evecs_pt
    

    
@cython.boundscheck(False)
@cython.cdivision(True)
cdef vec __sample_mu_Sigma(mat * mu, cube * Sigma, vec * labels, mat * data,
                           double gamma, vec * pr_mu, double pr_nu, mat * pr_phi,
                           vec * hdp_rngs, do_parallel=True, hdp=False):
    cdef bool is_hdp
    cdef int k, nthd, chunk, i, J, cnt
    cdef rng_sampler[double] * R = new rng_sampler[double]()
    cdef int p = mu.n_rows
    cdef int ncomp = mu.n_cols

    cdef vec * count = new vec(ncomp)
    cdef vec * hdp_count
    cdef double * hdp_count_ptr
    # hdp stuff ... 
    if hdp:
        J = hdp_rngs.n_elem
        hdp_count = new vec(ncomp*J)
        hdp_count_ptr = hdp_count.memptr()
        for i in range(ncomp*J):
            hdp_count_ptr[i] = 0
        cnt = 0
        is_hdp = True
    else:
        is_hdp = False
    cdef double * count_ptr = count.memptr()
    cdef double * labels_ptr = labels.memptr()

    cdef vector[vector[int]] * mask = new vector[vector[int]]()
    cdef vector[int] * cur_mask

    for k in range(ncomp):
        count_ptr[k]=0
        cur_mask = new vector[int]()
        mask.push_back(deref(cur_mask))
    #cdef vec count = deref(count_p)

    cdef vec * cmu
    cdef mat * cSigma

    if do_parallel:
        nthd = multiprocessing.cpu_count()
        chunk = 1
    else:
        nthd = 1
        chunk = ncomp

    for i in range(data.n_cols):
        k = <int>labels_ptr[i]
        count_ptr[k] += 1
        mask.at(k).push_back(i)
        if is_hdp:
            hdp_count_ptr[cnt*ncomp + k] += 1
            if i >= hdp_rngs.at(cnt):
                cnt += 1

    for k in prange(ncomp,nogil=True, num_threads=nthd, schedule='dynamic', chunksize=chunk):
        cSigma = cube_slice_view(Sigma, k)
        cmu = mat_col_view(mu,k)
        _sample_component(deref(cmu), deref(cSigma),
                          deref(data), deref(count)[k], deref(mask).at(k),
                          gamma, deref(pr_mu),
                          pr_nu, deref(pr_phi), deref(R))

    for k in range(ncomp):
        mask.erase(mask.begin())
    del mask
    del R

    if is_hdp:
        return deref(hdp_count)
    else:
        return deref(count)

def sample_mu_Sigma(np.ndarray[np.double_t, ndim=2] mu_in,
                    np.ndarray[np.double_t, ndim=3] Sigma_in,
                    np.ndarray[np.int_t, ndim=1] labels_in,
                    np.ndarray[np.double_t, ndim=2] data_in,
                    double gamma, np.ndarray[np.double_t, ndim=1] pr_mu_in,
                    double pr_nu, np.ndarray[np.double_t, ndim=2] pr_phi_in,
                    parallel=True, np.ndarray[np.double_t, ndim=1] hdp_rngs_in=None):

    cdef vec * hdp_rngs

    cdef int J, K

    if hdp_rngs_in is None:
        hdp = False
    else:
        hdp = True
        hdp_rngs = numpy_to_vec(hdp_rngs_in)
        J = hdp_rngs.n_elem
        K = mu_in.shape[0]

    ## this code just handles type casting
    ## will eventually handle hdp and parallel issues

    ## mu_in and Sigma_in are updated in place
    
    ## convert to armadillo
    cdef mat * mu = numpy_to_mat(mu_in.T)
    cdef cube * Sigma = numpy_to_cube(Sigma_in)
    cdef np.ndarray[np.double_t] labels_in_dbl = np.array(labels_in, dtype=np.double)
    cdef vec * labels = numpy_to_vec(labels_in_dbl)
    cdef mat * data = numpy_to_mat(data_in.T)
    cdef vec * pr_mu = numpy_to_vec(pr_mu_in)
    cdef mat * pr_phi = numpy_to_mat(pr_phi_in.T)
    cdef vec ct

    ## call
    if hdp:
        ct = __sample_mu_Sigma(mu, Sigma, labels, data,
                               gamma, pr_mu, pr_nu, pr_phi, hdp_rngs, parallel, True)
        return np.array(vec_to_numpy(ct, None), dtype=np.int).reshape(J,K)
    else:
        ct = __sample_mu_Sigma(mu, Sigma, labels, data,
                               gamma, pr_mu, pr_nu, pr_phi, hdp_rngs, parallel)
        return np.array(vec_to_numpy(ct,None),dtype=np.int)

############### HDP Samplers ######################

cdef inline double log_beta_pdf(double x, double a, double b) nogil:
    return lgamma(a+b) - lgamma(a) - lgamma(b) + (a-1)*log(x) + (b-1)*log(1-x)

cdef inline double log_gamma_pdf(double x, double alpha, double beta) nogil:
    return alpha*log(beta) - lgamma(alpha) + (alpha-1)*log(x) - beta*x

@cython.boundscheck(False)
@cython.cdivision(True)
cdef double beta_post(double* stick_beta,
                      double* beta,
                      double* stick_weights,
                      double  alpha0, double  alpha, int J, int K,
                      int sw_str_r, int sw_str_c, int pos,
                      double* likelihood, int nthreads) nogil:
    cdef double a, b
    cdef int j, k
#    cdef int J = stick_weights.shape[0]
#    cdef int k = beta.shape[0] (or ncomp)
    cdef double lpost
    cdef double cumsum = 0.0
    cdef double * beta_cumsum = <double*>calloc(K, sizeof(double))
    for k in range(K):
        cumsum += beta[k]
        beta_cumsum[k] = cumsum

    for k in prange(pos, K-1, schedule="static", num_threads=nthreads):
        likelihood[k] = 0.0
        #cumsum += beta[k]
        a = alpha0*beta[k]
        b = alpha0*(1-beta_cumsum[k])
        for j in range(J-1):
            likelihood[k] += log_beta_pdf(stick_weights[j*sw_str_r+k*sw_str_c], a, b)
        likelihood[k] += log_beta_pdf(stick_beta[k], 1.0, alpha)

    lpost = 0.0
    for k in range(0,K-1):
        lpost += likelihood[k]

    free(beta_cumsum)
    return lpost

@cython.boundscheck(False)
@cython.cdivision(True)
cdef np.ndarray[np.double_t] break_sticks(np.ndarray[np.double_t, ndim=1]  V):
    cdef int n = V.shape[0]
    cdef int k 
    cdef np.ndarray[np.double_t] pi = np.zeros(n+1, dtype=np.double)
    pi[0] = V[0]
    cdef double prod = (1-V[0])
    for k in range(1, n):
        pi[k] = prod * V[k]
        prod *= 1 - V[k]
    pi[-1] = prod
    return pi

@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline void update_beta(double * beta, double olen, double nlen, int pos, int k) nogil:
    cdef int i
    beta[pos] *= nlen / olen
    if pos < k-1:
        for i in range(pos+1,k):
            beta[i] *= (1.0 - nlen) / (1.0 - olen)

@cython.boundscheck(False)
@cython.cdivision(True)
def sample_beta(np.ndarray[np.double_t, ndim=1] stick_beta,
                np.ndarray[np.double_t, ndim=1] beta,
                np.ndarray[np.double_t, ndim=2] stick_weights,
                double alpha0, double alpha,
                np.ndarray[np.double_t, ndim=1] AR,
                np.ndarray[np.double_t, ndim=1] prop_scale,
                do_parallel = False):


    cdef rng_sampler[double] * R = new rng_sampler[double]()
    cdef np.ndarray[np.double_t] old_stick_beta = stick_beta.copy()
    cdef np.ndarray[np.double_t] old_beta = beta.copy()
    # to make it GIL free, get pointers to all array data        
    cdef double * old_stick_beta_ptr = &old_stick_beta[0]
    cdef double * old_beta_ptr = &old_beta[0]
    cdef double * beta_ptr = &beta[0]
    cdef double * stick_beta_ptr = &stick_beta[0]
    cdef double * stick_weights_ptr = &stick_weights[0,0]
    cdef double * prop_scale_ptr = &prop_scale[0]
    cdef double * AR_ptr = &AR[0]
    # misc
    cdef int ncomp = beta.shape[0]
    cdef int J = stick_weights.shape[0]
    cdef int str_r = stick_weights.strides[0] / stick_weights.itemsize
    cdef int str_c = stick_weights.strides[1] / stick_weights.itemsize
    cdef double * lpost_new = <double*>calloc(1,sizeof(double))
    cdef double * lpost = <double*>calloc(1,sizeof(double))
    cdef double * prop = <double*>calloc(1,sizeof(double))
    # likelihood working variable
    cdef double * like_work = <double*>calloc(ncomp, sizeof(double))

    # get initial logpost
    cdef int nthreads = 1
    if do_parallel:
        nthreads = multiprocessing.cpu_count()
    openmp.omp_set_nested(1)
    cdef int tid, k
    with nogil, parallel(num_threads = 1):
        tid = threadid()
        lpost[0] = beta_post(stick_beta_ptr, beta_ptr,
                          stick_weights_ptr, alpha0, alpha,
                          J, ncomp, str_r, str_c, 0, like_work, nthreads)
        for k in range(ncomp-1):
        
            # sample new beta from reflected normal
            #prop = stats.norm.rvs(stick_beta[k], self.prop_scale[k])
            if tid == 0:
                prop[0] = R.normal(stick_beta_ptr[k], prop_scale_ptr[k])
                while prop[0] > (1-1e-9) or prop[0] < 1e-9:
                    if prop[0] > 1-1e-9:
                        prop[0] = 2*(1-1e-9) - prop[0]
                    else:
                        prop[0] = 2*1e-9 - prop[0]
                update_beta(beta_ptr, stick_beta_ptr[k], prop[0], k, ncomp) #beta = break_sticks(stick_beta)
                stick_beta_ptr[k] = prop[0]
                #OMP_BARRIER 

                # get new posterior
                lpost_new[0] = beta_post(stick_beta_ptr, beta_ptr,
                                         stick_weights_ptr, alpha0, alpha,
                                         J, ncomp, str_r, str_c, k, like_work, nthreads)
                #if tid == 0:
                # accept or reject
                if R.exp(1.0) > lpost[0] - lpost_new[0]:
                    #accept
                    AR_ptr[k] += 1
                    lpost[0] = lpost_new[0]
                else:
                    update_beta(beta_ptr, stick_beta_ptr[k], old_stick_beta_ptr[k], k, ncomp)
                    #beta = break_sticks(stick_beta)
                    stick_beta_ptr[k] = old_stick_beta_ptr[k]
                #OMP_BARRIER 

    del R
    beta = break_sticks(stick_beta) # help avoid rounding errors
    free(like_work)
    free(prop)
    free(lpost)
    free(lpost_new)
    return stick_beta, beta

@cython.boundscheck(False)
@cython.cdivision(True)
def sample_alpha0(np.ndarray[np.double_t, ndim=2] stick_weights,
                  np.ndarray[np.double_t, ndim=1] beta, double alpha0,
                  double e0, double f0,
                  np.ndarray[np.double_t, ndim=1] prop_scale,
                  np.ndarray[np.double_t, ndim=1] AR):
    # just reuse with dummy vars for beta things
    cdef rng_sampler[double] * R = new rng_sampler[double]()
    cdef int ncomp = beta.shape[0]
    cdef int J = stick_weights.shape[0]
    cdef int str_r = stick_weights.strides[0] / stick_weights.itemsize
    cdef int str_c = stick_weights.strides[1] / stick_weights.itemsize
    cdef np.ndarray[np.double_t] tmp_ones = 0.5*np.ones_like(beta)

    cdef double* tmp = <double*>calloc(ncomp, sizeof(double))
    cdef double lpost = beta_post(&tmp_ones[0], &beta[0], &stick_weights[0,0], alpha0, 1,
                                  J, ncomp, str_r, str_c, 0, tmp, 0)
    lpost += log_gamma_pdf(alpha0, e0, f0)
    cdef double alpha0_old = alpha0
    alpha0 = fabs(R.normal(alpha0, prop_scale[-1]))
    cdef double lpost_new = beta_post(&tmp_ones[0], &beta[0], &stick_weights[0,0], alpha0, 1,
                                      J, ncomp, str_r, str_c, 0, tmp, 0)
    lpost_new += log_gamma_pdf(alpha0, e0, f0)
    #accept or reject
    if R.exp(1) > lpost - lpost_new:
        AR[-1] += 1
    else:
        alpha0 = alpha0_old
    del R
    free(tmp)
    return alpha0

# ############ some python functions for testing ############

def pyiwishartrand(double nu, np.ndarray[np.double_t, ndim=2] Phi):
    #cdef rng r
    cdef rng_sampler[double] * R = new rng_sampler[double]()
    cdef mat result = invwishartrand(nu, deref(numpy_to_mat(Phi)), deref(R))
    del R
    return mat_to_numpy(result, None)

def pyiwishartrand_prec(double nu, np.ndarray[np.double_t, ndim=2] Phi):
    #cdef rng r
    cdef rng_sampler[double] * R = new rng_sampler[double]()
    cdef mat result = invwishartrand_prec(nu, deref(numpy_to_mat(Phi)), deref(R))
    del R
    return mat_to_numpy(result, None)

def pymvnorm(np.ndarray[np.double_t, ndim=1] mu,
              np.ndarray[np.double_t, ndim=2] Sigma):

    cdef rng_sampler[double] * R = new rng_sampler[double]()
    cdef vec result = mvnormrand(deref(numpy_to_vec(mu)), deref(numpy_to_mat(Sigma)), deref(R))
    del R
    return vec_to_numpy(result, None)

def pymvnorm_prec(np.ndarray[np.double_t, ndim=1] mu,
              np.ndarray[np.double_t, ndim=2] Sigma):

    cdef rng_sampler[double] * R = new rng_sampler[double]()
    cdef vec result = mvnormrand_prec(deref(numpy_to_vec(mu)),
                                      deref(numpy_to_mat(Sigma)), deref(R))
    del R
    return vec_to_numpy(result, None)




